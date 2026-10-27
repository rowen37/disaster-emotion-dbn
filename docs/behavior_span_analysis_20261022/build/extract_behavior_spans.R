args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: Rscript extract_behavior_spans.R <data_stage2_dir> <output_csv>")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_csv <- args[[2]]
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)

hazard_labels <- c(
  debris_flow = "泥石流",
  earthquake = "地震",
  landslide = "滑坡",
  typhoon = "台风"
)

frames <- lapply(names(hazard_labels), function(hazard) {
  path <- file.path(input_dir, paste0(hazard, ".rds"))
  frame <- readRDS(path)
  required <- c("text", "event_id", "has_behavior", "behavior_spans")
  missing <- setdiff(required, names(frame))
  if (length(missing)) {
    stop(sprintf("%s missing required columns: %s", path, paste(missing, collapse = ", ")))
  }

  fallback <- if ("behavior_span_fallback" %in% names(frame)) {
    frame$behavior_span_fallback
  } else {
    rep(FALSE, nrow(frame))
  }

  data.frame(
    hazard = hazard,
    hazard_cn = unname(hazard_labels[[hazard]]),
    source_row = seq_len(nrow(frame)),
    event_id = as.character(frame$event_id),
    text = as.character(frame$text),
    has_behavior = as.logical(frame$has_behavior),
    behavior_spans = as.character(frame$behavior_spans),
    behavior_span_fallback = as.logical(fallback),
    stringsAsFactors = FALSE
  )
})

all_rows <- do.call(rbind, frames)
write.csv(all_rows, output_csv, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("wrote %s rows to %s\n", nrow(all_rows), output_csv))
