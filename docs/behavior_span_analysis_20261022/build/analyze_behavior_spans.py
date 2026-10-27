#!/usr/bin/env python3
"""Create de-duplicated word-frequency and per-hazard LDA tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import re
import unicodedata

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


HAZARD_ORDER = ["debris_flow", "earthquake", "landslide", "typhoon"]
HAZARD_CN = {
    "debris_flow": "泥石流",
    "earthquake": "地震",
    "landslide": "滑坡",
    "typhoon": "台风",
}

# Function words, conversational fillers, time words and disaster labels are removed
# so the tables foreground actions rather than the surrounding event context.
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "也", "有", "就", "都", "不", "人", "与", "及", "或",
    "而", "但", "被", "把", "让", "给", "向", "从", "到", "去", "来", "为", "于", "上", "下",
    "中", "里", "外", "后", "前", "又", "还", "很", "太", "更", "最", "再", "已", "要", "会",
    "能", "可", "得", "着", "过", "这", "那", "你", "他", "她", "它", "们", "其", "之", "者",
    "啊", "呀", "呢", "吧", "吗", "哦", "嗯", "哈", "啦", "么", "嘛", "哇", "喽", "呗",
    "一个", "一些", "一种", "一样", "一下", "一次", "一点", "一边", "一起", "一切", "一会儿",
    "这个", "那个", "这些", "那些", "这样", "那样", "这种", "那种", "这里", "那里", "其中",
    "什么", "怎么", "为什么", "多少", "哪儿", "哪个", "如何", "是否", "如果", "假如", "因为",
    "所以", "但是", "而且", "或者", "以及", "并且", "然后", "不过", "虽然", "已经", "还是",
    "可以", "可能", "应该", "需要", "必须", "能够", "不能", "不会", "没有", "不是", "就是",
    "自己", "大家", "有人", "别人", "我们", "你们", "他们", "本人", "东西", "事情", "情况",
    "时候", "时间", "地方", "方面", "问题", "感觉", "觉得", "发现", "看到", "知道", "听说",
    "表示", "进行", "开始", "继续", "一直", "正在", "直接", "突然", "赶紧", "马上", "终于",
    "目前", "现在", "当时", "今天", "昨天", "明天", "晚上", "早上", "上午", "下午", "凌晨",
    "今年", "去年", "每天", "最近", "当年", "期间", "之前", "之后", "以来", "现场", "当地",
    "真的", "特别", "非常", "比较", "这么", "那么", "这么多", "好多", "各种", "相关", "由于",
    "通过", "对于", "关于", "为了", "为了", "同时", "甚至", "仍然", "依然", "完全", "基本",
    "出现", "造成", "导致", "发生", "发布", "视频", "微博", "消息", "信息", "网友", "网民",
    "台风", "地震", "滑坡", "泥石流", "灾害", "受灾", "暴雨", "大雨", "风雨", "天气",
    "灾区", "汛期", "山体", "山洪", "震区", "余震", "登陆", "影响", "紧急", "提醒",
    "说", "说是", "没事", "的话", "一下子", "真是", "竟然", "居然", "原来", "结果",
}

CUSTOM_WORDS = [
    "自驾", "撤离", "转移", "疏散", "抢购", "囤货", "备货", "停课", "停工", "停业", "停航",
    "停运", "返程", "出行", "救援", "抢险", "捐款", "捐物", "造谣", "辟谣", "插队", "违停",
    "闯红灯", "买菜", "回家", "离家", "逃离", "报警", "求救", "上报", "请假", "取消行程",
    "重新规划", "封路", "封闭", "加固", "关窗", "改签", "退票", "绕行", "断电", "断水",
    "志愿服务", "运送物资", "发放物资", "搭建帐篷", "清理道路", "抢修", "避险", "避难",
]

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"@[\w\-\u4e00-\u9fff]+")
TOKEN_PIECE_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    return SPACE_RE.sub("", normalize_text(text)).lower()


def parse_spans(raw: object) -> tuple[list[str], str]:
    text = normalize_text(raw)
    if not text or text == "[]":
        return [], "empty"
    candidates = [(text, "direct")]
    if '""' in text:
        candidates.append((text.replace('""', '"'), "double_quote_repair"))
    for candidate, mode in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        spans = [normalize_text(item) for item in value]
        return [item for item in spans if item], mode
    return [], "parse_error"


def tokenize_span(span: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in jieba.lcut(normalize_text(span), cut_all=False, HMM=True):
        for piece in TOKEN_PIECE_RE.findall(raw_token):
            token = piece.lower()
            if token in STOPWORDS:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]", token):
                continue
            if token.isascii() and len(token) < 2:
                continue
            tokens.append(token)
    return tokens


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument("--topics", type=int, default=5)
    parser.add_argument("--top-words", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    for word in CUSTOM_WORDS:
        jieba.add_word(word, freq=100000)

    frame = pd.read_csv(args.input_csv, dtype={"event_id": "string"}, keep_default_na=False)
    frame["has_behavior"] = frame["has_behavior"].astype(str).str.upper().eq("TRUE")
    frame["behavior_span_fallback"] = (
        frame["behavior_span_fallback"].astype(str).str.upper().eq("TRUE")
    )

    parsed: list[list[str]] = []
    modes: list[str] = []
    for raw in frame["behavior_spans"].tolist():
        spans, mode = parse_spans(raw)
        parsed.append(spans)
        modes.append(mode)
    frame["spans"] = parsed
    frame["parse_mode"] = modes
    frame["text_key"] = frame["text"].map(dedup_key)

    summaries: list[dict] = []
    frequencies: list[dict] = []
    lda_topics: list[dict] = []
    lda_terms: list[dict] = []
    top_words_by_hazard: dict[str, list[str]] = {}

    for hazard in HAZARD_ORDER:
        all_hazard = frame.loc[frame["hazard"] == hazard].copy()
        positive = all_hazard.loc[
            all_hazard["has_behavior"] & all_hazard["spans"].map(bool)
        ].copy()

        # Exact normalized text is the post identity. If duplicated rows disagree on
        # spans, retain the union of extracted spans for the unique-post document.
        docs: list[dict] = []
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in positive.iterrows():
            key = row["text_key"] or f"__blank__{int(row['source_row'])}"
            groups[key].append(idx)
        for indices in groups.values():
            rows = positive.loc[indices]
            spans = ordered_unique([s for row_spans in rows["spans"] for s in row_spans])
            tokens = [token for span in spans for token in tokenize_span(span)]
            docs.append(
                {
                    "spans": spans,
                    "tokens": tokens,
                    "fallback": bool(rows["behavior_span_fallback"].any()),
                }
            )

        raw_tokens = [
            token
            for row_spans in positive["spans"]
            for span in row_spans
            for token in tokenize_span(span)
        ]
        clean_docs = [doc for doc in docs if doc["tokens"]]
        clean_tokens = [token for doc in clean_docs for token in doc["tokens"]]
        raw_counts = Counter(raw_tokens)
        dedup_counts = Counter(clean_tokens)
        doc_counts = Counter(
            token for doc in clean_docs for token in set(doc["tokens"])
        )

        ordered_terms = sorted(dedup_counts, key=lambda t: (-dedup_counts[t], -doc_counts[t], t))
        for rank, term in enumerate(ordered_terms, start=1):
            frequencies.append(
                {
                    "灾害": HAZARD_CN[hazard],
                    "灾害代码": hazard,
                    "排名": rank,
                    "词语": term,
                    "去重词频": int(dedup_counts[term]),
                    "原始行词频": int(raw_counts[term]),
                    "文档频数": int(doc_counts[term]),
                    "文档占比": float(doc_counts[term] / len(clean_docs)) if clean_docs else 0.0,
                }
            )
        top_words_by_hazard[HAZARD_CN[hazard]] = ordered_terms[: args.top_words]

        model_docs = [doc["tokens"] for doc in clean_docs]
        parse_error_rows = int((all_hazard["parse_mode"] == "parse_error").sum())
        repaired_rows = int((all_hazard["parse_mode"] == "double_quote_repair").sum())
        topic_count = 0
        vocab_size = 0
        perplexity = None
        log_likelihood = None

        if model_docs:
            lda_min_df = max(2, int(math.ceil(len(model_docs) * 0.002)))
            vectorizer = CountVectorizer(
                analyzer=lambda tokens: tokens,
                lowercase=False,
                token_pattern=None,
                min_df=lda_min_df,
                max_df=0.95,
                max_features=3000,
            )
            matrix = vectorizer.fit_transform(model_docs)
            vocab = np.asarray(vectorizer.get_feature_names_out())
            vocab_size = int(len(vocab))
            if vocab_size >= 10 and matrix.shape[0] >= args.topics:
                topic_count = args.topics
                model = LatentDirichletAllocation(
                    n_components=topic_count,
                    random_state=args.random_state,
                    learning_method="batch",
                    max_iter=50,
                    evaluate_every=-1,
                    n_jobs=1,
                )
                doc_topic = model.fit_transform(matrix)
                topic_share = doc_topic.mean(axis=0)
                perplexity = float(model.perplexity(matrix))
                log_likelihood = float(model.score(matrix))
                normalized_components = model.components_ / model.components_.sum(axis=1, keepdims=True)

                for topic_idx in range(topic_count):
                    top_indices = np.argsort(normalized_components[topic_idx])[::-1][: args.top_words]
                    top_terms = [str(vocab[i]) for i in top_indices]
                    top_term_counts = np.asarray(matrix[:, top_indices].sum(axis=1)).ravel()
                    representative_score = doc_topic[:, topic_idx] * (1.0 + top_term_counts)
                    representative_indices = np.argsort(representative_score)[::-1]
                    representative_spans: list[str] = []
                    for doc_idx in representative_indices:
                        if top_term_counts[int(doc_idx)] <= 0:
                            continue
                        phrase = "；".join(clean_docs[int(doc_idx)]["spans"])
                        if phrase and phrase not in representative_spans:
                            representative_spans.append(phrase[:120])
                        if len(representative_spans) == 3:
                            break
                    lda_topics.append(
                        {
                            "灾害": HAZARD_CN[hazard],
                            "灾害代码": hazard,
                            "主题": topic_idx + 1,
                            "主题占比": float(topic_share[topic_idx]),
                            "Top10主题词": "、".join(top_terms),
                            "代表Span 1": representative_spans[0] if len(representative_spans) > 0 else "",
                            "代表Span 2": representative_spans[1] if len(representative_spans) > 1 else "",
                            "代表Span 3": representative_spans[2] if len(representative_spans) > 2 else "",
                        }
                    )
                    for rank, term_idx in enumerate(top_indices, start=1):
                        lda_terms.append(
                            {
                                "灾害": HAZARD_CN[hazard],
                                "灾害代码": hazard,
                                "主题": topic_idx + 1,
                                "排名": rank,
                                "词语": str(vocab[term_idx]),
                                "主题内权重": float(normalized_components[topic_idx, term_idx]),
                            }
                        )

        raw_span_count = int(sum(len(spans) for spans in positive["spans"]))
        dedup_span_count = int(sum(len(doc["spans"]) for doc in docs))
        summaries.append(
            {
                "灾害": HAZARD_CN[hazard],
                "灾害代码": hazard,
                "总帖子行数": int(len(all_hazard)),
                "行为Span帖子行数": int(len(positive)),
                "唯一帖子数": int(len(docs)),
                "重复行占比": float(1 - len(docs) / len(positive)) if len(positive) else 0.0,
                "原始Span数": raw_span_count,
                "去重Span数": dedup_span_count,
                "清洗后文档数": int(len(clean_docs)),
                "词表规模": int(len(dedup_counts)),
                "LDA词表规模": vocab_size,
                "LDA最小文档频数": int(max(2, math.ceil(len(model_docs) * 0.002))) if model_docs else 0,
                "LDA主题数": topic_count,
                "困惑度": perplexity,
                "对数似然": log_likelihood,
                "修复双引号行数": repaired_rows,
                "解析失败行数": parse_error_rows,
                "fallback唯一帖子数": int(sum(doc["fallback"] for doc in docs)),
            }
        )

    result = {
        "summary": summaries,
        "frequencies": frequencies,
        "lda_topics": lda_topics,
        "lda_terms": lda_terms,
        "top_words": top_words_by_hazard,
        "method": {
            "source": "data_stage2/debris_flow.rds, earthquake.rds, landslide.rds, typhoon.rds",
            "document_unit": "同一灾害内按NFKC规范化、去除空白后的完整帖子文本精确去重；重复行的行为Span取并集",
            "word_frequency": "仅按去重后的唯一帖子统计；词频为唯一帖子全部行为Span中的出现次数，文档频数为含该词的唯一帖子数",
            "tokenization": "jieba 0.42.1 精确模式；NFKC规范化；移除URL、@提及、数字、标点、单字词、停用词和灾害名称",
            "lda": (
                f"每个灾害单独建模；帖子内全部行为Span合并为一个文档；"
                f"CountVectorizer(min_df=max(2, 向上取整(文档数×0.2%)), max_df=0.95, max_features=3000)；"
                f"sklearn LDA K={args.topics}, batch, max_iter=50, random_state={args.random_state}"
            ),
            "stopwords": sorted(STOPWORDS),
            "custom_words": CUSTOM_WORDS,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
