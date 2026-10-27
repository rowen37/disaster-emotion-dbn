import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const buildDir = path.resolve("outputs/behavior_span_analysis_20261022/build");
const outputDir = path.resolve("outputs/behavior_span_analysis_20261022");
const resultPath = path.join(buildDir, "analysis_results.json");
const outputPath = path.join(outputDir, "behavior_span_wordfreq_lda.xlsx");
const previewDir = path.join(buildDir, "previews");
const data = JSON.parse(await fs.readFile(resultPath, "utf8"));

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const summary = workbook.worksheets.add("摘要");
const wordFreq = workbook.worksheets.add("词频明细");
const ldaTopics = workbook.worksheets.add("LDA主题");
const ldaTerms = workbook.worksheets.add("LDA主题词");
const methods = workbook.worksheets.add("方法说明");

const FONT = "Arial";
const COLORS = {
  navy: "#1F4E78",
  blue: "#5B9BD5",
  lightBlue: "#D9EAF7",
  paleBlue: "#EEF5FB",
  text: "#1F2937",
  muted: "#5F6B7A",
  line: "#D6DEE6",
  lightGray: "#F3F5F7",
  amber: "#FFF2CC",
  red: "#FCE8E6",
};

function setBase(sheet, rangeAddress) {
  sheet.showGridLines = false;
  const range = sheet.getRange(rangeAddress);
  range.format.font = { name: FONT, size: 10, color: COLORS.text };
  range.format.verticalAlignment = "center";
}

function setTitle(sheet, title, subtitle, lastCol) {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: FONT, size: 16, bold: true, color: COLORS.navy };
  sheet.getRange(`A3:${lastCol}3`).format.borders = {
    bottom: { style: "thin", color: COLORS.blue },
  };
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3").format.font = { name: FONT, size: 10, italic: true, color: COLORS.muted };
}

function setSection(sheet, cell, text, endCell) {
  sheet.getRange(cell).values = [[text]];
  const range = sheet.getRange(`${cell}:${endCell}`);
  range.format.fill = COLORS.lightBlue;
  range.format.font = { name: FONT, size: 10, bold: true, color: COLORS.navy };
  range.format.borders = { preset: "outside", style: "thin", color: COLORS.line };
}

function styleHeader(range) {
  range.format.fill = COLORS.navy;
  range.format.font = { name: FONT, size: 10, bold: true, color: "#FFFFFF" };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.borders = {
    insideVertical: { style: "thin", color: "#FFFFFF" },
    bottom: { style: "thin", color: COLORS.navy },
  };
  range.format.rowHeight = 30;
}

function styleDataArea(range) {
  range.format.borders = {
    insideHorizontal: { style: "thin", color: COLORS.line },
    bottom: { style: "thin", color: COLORS.line },
  };
}

function writeMatrix(sheet, startRow, startCol, rows) {
  if (!rows.length) return null;
  const range = sheet.getRangeByIndexes(startRow, startCol, rows.length, rows[0].length);
  range.values = rows;
  return range;
}

// Summary
setBase(summary, "A1:L26");
setTitle(
  summary,
  "四类灾害行为 Span：词频与 LDA",
  "词频和 LDA 均按灾害内唯一帖子计算；原始行数仅用于记录去重规模。",
  "L",
);
setSection(summary, "A5", "数据概览", "L5");
const summaryHeaders = [
  "灾害", "原始帖子行数", "原始行为Span行数", "唯一行为帖子数", "重复行占比", "原始Span数",
  "去重Span数", "清洗后文档数", "词表规模", "LDA词表规模", "LDA主题数", "fallback唯一帖子数",
];
const summaryRows = data.summary.map((row) => [
  row["灾害"], row["总帖子行数"], row["行为Span帖子行数"], row["唯一帖子数"], row["重复行占比"],
  row["原始Span数"], row["去重Span数"], row["清洗后文档数"], row["词表规模"], row["LDA词表规模"],
  row["LDA主题数"], row["fallback唯一帖子数"],
]);
writeMatrix(summary, 5, 0, [summaryHeaders, ...summaryRows]);
styleHeader(summary.getRange("A6:L6"));
styleDataArea(summary.getRange("A7:L10"));
summary.getRange("B7:D10").format.numberFormat = "#,##0";
summary.getRange("E7:E10").format.numberFormat = "0.0%";
summary.getRange("F7:L10").format.numberFormat = "#,##0";
summary.tables.add("A6:L10", true, "SummaryTable");

setSection(summary, "A13", "各灾害唯一帖子词频 Top 10", "K13");
const topHeaders = ["灾害", ...Array.from({ length: 10 }, (_, i) => `第${i + 1}名`)];
const topRows = ["泥石流", "地震", "滑坡", "台风"].map((hazard) => [
  hazard,
  ...Array.from({ length: 10 }, (_, i) => data.top_words[hazard]?.[i] ?? ""),
]);
writeMatrix(summary, 13, 0, [topHeaders, ...topRows]);
styleHeader(summary.getRange("A14:K14"));
styleDataArea(summary.getRange("A15:K18"));
summary.getRange("B15:K18").format.horizontalAlignment = "center";

setSection(summary, "A21", "关键处理口径", "L21");
const processingRows = [
  ["去重", "灾害内按唯一帖子计算"],
  ["分词", "jieba 精确模式；移除单字词和停用词"],
  ["LDA", "每类灾害单独建立 5 个主题"],
  ["数据质量", "台风引号已修复；解析失败 0 行"],
];
writeMatrix(summary, 21, 0, [["处理", "说明"], ...processingRows]);
styleHeader(summary.getRange("A22:B22"));
styleDataArea(summary.getRange("A23:B26"));
summary.getRange("B23:B26").format.wrapText = true;
summary.getRange("B23:B26").format.rowHeight = 34;
summary.getRange("A1:A26").format.columnWidth = 16;
summary.getRange("B1:B26").format.columnWidth = 45;
summary.getRange("C1:L26").format.columnWidth = 14;
summary.getRange("B14:K18").format.columnWidth = 15;
summary.tabColor = COLORS.navy;

// Word frequencies
const freqHeaders = ["灾害", "灾害代码", "排名", "词语", "唯一帖子词频", "文档频数", "文档占比"];
// Rename the analysis key for the user-facing table; all counts here use unique posts.
for (const row of data.frequencies) row["唯一帖子词频"] = row["去重词频"];
const uniqueFreqRows = data.frequencies.map((row) => freqHeaders.map((key) => row[key]));
const freqEndRow = 5 + uniqueFreqRows.length;
setBase(wordFreq, `A1:G${freqEndRow}`);
setTitle(
  wordFreq,
  "行为 Span 词频明细",
  "同一灾害内按唯一帖子计算；词频为全部行为 Span 中的出现次数，文档频数为含该词的唯一帖子数。",
  "G",
);
writeMatrix(wordFreq, 4, 0, [freqHeaders, ...uniqueFreqRows]);
styleHeader(wordFreq.getRange("A5:G5"));
styleDataArea(wordFreq.getRange(`A6:G${freqEndRow}`));
wordFreq.getRange(`C6:C${freqEndRow}`).format.numberFormat = "#,##0";
wordFreq.getRange(`E6:F${freqEndRow}`).format.numberFormat = "#,##0";
wordFreq.getRange(`G6:G${freqEndRow}`).format.numberFormat = "0.00%";
wordFreq.tables.add(`A5:G${freqEndRow}`, true, "WordFrequencyTable");
wordFreq.freezePanes.freezeRows(5);
wordFreq.freezePanes.freezeColumns(2);
wordFreq.getRange(`A1:A${freqEndRow}`).format.columnWidth = 12;
wordFreq.getRange(`B1:B${freqEndRow}`).format.columnWidth = 16;
wordFreq.getRange(`C1:C${freqEndRow}`).format.columnWidth = 9;
wordFreq.getRange(`D1:D${freqEndRow}`).format.columnWidth = 22;
wordFreq.getRange(`E1:G${freqEndRow}`).format.columnWidth = 16;
wordFreq.tabColor = COLORS.blue;

// LDA topic summaries
const topicHeaders = ["灾害", "灾害代码", "主题", "主题占比", "Top10主题词", "代表Span 1", "代表Span 2", "代表Span 3"];
const topicRows = data.lda_topics.map((row) => topicHeaders.map((key) => row[key]));
const topicEndRow = 5 + topicRows.length;
setBase(ldaTopics, `A1:H${topicEndRow}`);
setTitle(
  ldaTopics,
  "LDA 主题汇总",
  "主题占比为唯一帖子在该主题上的平均后验概率；代表 Span 优先选择含 Top 词且主题概率较高的帖子。",
  "H",
);
writeMatrix(ldaTopics, 4, 0, [topicHeaders, ...topicRows]);
styleHeader(ldaTopics.getRange("A5:H5"));
styleDataArea(ldaTopics.getRange(`A6:H${topicEndRow}`));
ldaTopics.getRange(`C6:C${topicEndRow}`).format.numberFormat = "0";
ldaTopics.getRange(`D6:D${topicEndRow}`).format.numberFormat = "0.0%";
ldaTopics.getRange(`E6:H${topicEndRow}`).format.wrapText = true;
ldaTopics.getRange(`E6:H${topicEndRow}`).format.verticalAlignment = "top";
ldaTopics.getRange(`A6:H${topicEndRow}`).format.rowHeight = 58;
ldaTopics.tables.add(`A5:H${topicEndRow}`, true, "LdaTopicTable");
ldaTopics.freezePanes.freezeRows(5);
ldaTopics.freezePanes.freezeColumns(2);
ldaTopics.getRange(`A1:A${topicEndRow}`).format.columnWidth = 12;
ldaTopics.getRange(`B1:B${topicEndRow}`).format.columnWidth = 16;
ldaTopics.getRange(`C1:C${topicEndRow}`).format.columnWidth = 8;
ldaTopics.getRange(`D1:D${topicEndRow}`).format.columnWidth = 12;
ldaTopics.getRange(`E1:E${topicEndRow}`).format.columnWidth = 48;
ldaTopics.getRange(`F1:H${topicEndRow}`).format.columnWidth = 42;
ldaTopics.tabColor = COLORS.blue;

// LDA term weights
const termHeaders = ["灾害", "灾害代码", "主题", "排名", "词语", "主题内权重"];
const termRows = data.lda_terms.map((row) => termHeaders.map((key) => row[key]));
const termEndRow = 5 + termRows.length;
setBase(ldaTerms, `A1:F${termEndRow}`);
setTitle(
  ldaTerms,
  "LDA 主题词权重",
  "每个主题列出权重最高的 10 个词；主题内权重为该词在主题词分布中的概率。",
  "F",
);
writeMatrix(ldaTerms, 4, 0, [termHeaders, ...termRows]);
styleHeader(ldaTerms.getRange("A5:F5"));
styleDataArea(ldaTerms.getRange(`A6:F${termEndRow}`));
ldaTerms.getRange(`C6:D${termEndRow}`).format.numberFormat = "0";
ldaTerms.getRange(`F6:F${termEndRow}`).format.numberFormat = "0.0000%";
ldaTerms.tables.add(`A5:F${termEndRow}`, true, "LdaTermTable");
ldaTerms.freezePanes.freezeRows(5);
ldaTerms.freezePanes.freezeColumns(2);
ldaTerms.getRange(`A1:A${termEndRow}`).format.columnWidth = 12;
ldaTerms.getRange(`B1:B${termEndRow}`).format.columnWidth = 16;
ldaTerms.getRange(`C1:D${termEndRow}`).format.columnWidth = 9;
ldaTerms.getRange(`E1:E${termEndRow}`).format.columnWidth = 22;
ldaTerms.getRange(`F1:F${termEndRow}`).format.columnWidth = 15;
ldaTerms.tabColor = COLORS.blue;

// Methods and diagnostics
setBase(methods, "A1:H34");
setTitle(methods, "方法与数据质量", "分析对象为模型抽取的行为 Span；结果用于探索性主题归纳。", "H");
setSection(methods, "A5", "分析方法", "H5");
const methodRows = [
  ["数据源", data.method.source],
  ["文档单位", data.method.document_unit],
  ["词频口径", data.method.word_frequency],
  ["分词与清洗", data.method.tokenization],
  ["LDA参数", data.method.lda],
  ["输出日期", "2026-10-22"],
];
writeMatrix(methods, 5, 0, [["项目", "说明"], ...methodRows]);
styleHeader(methods.getRange("A6:B6"));
styleDataArea(methods.getRange("A7:B12"));
methods.getRange("B7:B12").format.wrapText = true;
methods.getRange("B7:B12").format.rowHeight = 36;

setSection(methods, "A15", "LDA 诊断与解析记录", "H15");
const diagHeaders = ["灾害", "LDA最小文档频数", "LDA词表规模", "主题数", "困惑度", "对数似然", "修复双引号行数", "解析失败行数"];
const diagRows = data.summary.map((row) => diagHeaders.map((key) => row[key]));
writeMatrix(methods, 15, 0, [diagHeaders, ...diagRows]);
styleHeader(methods.getRange("A16:H16"));
styleDataArea(methods.getRange("A17:H20"));
methods.getRange("B17:D20").format.numberFormat = "#,##0";
methods.getRange("E17:F20").format.numberFormat = "#,##0.00";
methods.getRange("G17:H20").format.numberFormat = "#,##0";

setSection(methods, "A23", "解释限制", "H23");
const limitationRows = [
  ["模型输出", "behavior_spans 为模型抽取结果，不等同于人工核验的真实行为标签。"],
  ["短片段", "原始数据含较多单字片段；分词后移除全部单字词，以降低主题噪声。"],
  ["重复记录", "台风数据重复比例较高；词频和 LDA 均按唯一帖子计算，重复行不参与重复计数。"],
  ["主题解释", "LDA 是探索性无监督模型；主题编号无序，主题名称应结合主题词和代表 Span 人工解释。"],
];
writeMatrix(methods, 23, 0, [["限制", "说明"], ...limitationRows]);
styleHeader(methods.getRange("A24:B24"));
styleDataArea(methods.getRange("A25:B28"));
methods.getRange("B25:B28").format.wrapText = true;
methods.getRange("B25:B28").format.rowHeight = 36;

setSection(methods, "A31", "词典设置", "H31");
const dictionaryRows = [
  ["停用词", data.method.stopwords.join("、")],
  ["自定义词", data.method.custom_words.join("、")],
];
writeMatrix(methods, 31, 0, [["类型", "内容"], ...dictionaryRows]);
styleHeader(methods.getRange("A32:B32"));
styleDataArea(methods.getRange("A33:B34"));
methods.getRange("B33:B34").format.wrapText = true;
methods.getRange("B33").format.rowHeight = 160;
methods.getRange("B34").format.rowHeight = 90;
methods.getRange("A1:A34").format.columnWidth = 18;
methods.getRange("B1:B34").format.columnWidth = 75;
methods.getRange("C1:H34").format.columnWidth = 16;
methods.tabColor = "#7F8C8D";

workbook.recalculate();

const checks = [];
checks.push((await workbook.inspect({
  kind: "table",
  range: "摘要!A1:L26",
  include: "values,formulas",
  tableMaxRows: 26,
  tableMaxCols: 12,
})).ndjson);
checks.push((await workbook.inspect({
  kind: "table",
  range: "LDA主题!A1:H12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8,
})).ndjson);
checks.push((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
})).ndjson);

const previews = [
  ["摘要", "A1:L26", "summary.png"],
  ["词频明细", "A1:G30", "word_frequency.png"],
  ["LDA主题", "A1:H25", "lda_topics.png"],
  ["LDA主题词", "A1:F35", "lda_terms.png"],
  ["方法说明", "A1:H34", "methods.png"],
];
for (const [sheetName, range, filename] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const savedBlob = await FileBlob.load(outputPath);
const reopened = await SpreadsheetFile.importXlsx(savedBlob);
const reopenedSummary = await reopened.inspect({
  kind: "table",
  range: "摘要!A5:L18",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 12,
});

console.log(JSON.stringify({
  outputPath,
  frequencyRows: uniqueFreqRows.length,
  ldaTopicRows: topicRows.length,
  ldaTermRows: termRows.length,
  checks,
  reopenedSummary: reopenedSummary.ndjson,
}, null, 2));
