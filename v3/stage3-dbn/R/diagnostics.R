# =====================================================================
# Collinearity diagnostics for the optimized Stage 3 main model.
# The design is reduced to one row per event x ring so the 20 repeated time
# bins do not artificially inflate the apparent sample size.
# =====================================================================

collinearity_design <- function(dat, cfg) {
  cov_names <- c(cfg$ev_covs, CONTEXT_COVARIATES)
  missing <- setdiff(cov_names, names(dat))
  if (length(missing) > 0) {
    stop("collinearity design missing: ", paste(missing, collapse = ", "))
  }
  x <- dat %>%
    dplyr::select(event_id, dist_bin, dplyr::all_of(cov_names)) %>%
    dplyr::distinct() %>%
    dplyr::select(dplyr::all_of(cov_names)) %>%
    as.data.frame()
  for (name in names(x)) x[[name]] <- suppressWarnings(as.numeric(x[[name]]))
  complete <- stats::complete.cases(x)
  x <- x[complete, , drop = FALSE]
  varying <- vapply(x, function(value) stats::sd(value) > 1e-8, logical(1))
  x <- x[, varying, drop = FALSE]
  if (nrow(x) < 3 || ncol(x) < 1) stop("insufficient complete covariate rows")
  list(x = x, n_complete = nrow(x), requested = cov_names)
}

compute_vif <- function(x) {
  if (ncol(x) == 1) {
    return(data.frame(variable = names(x), vif = 1, tolerance = 1))
  }
  rows <- lapply(seq_along(x), function(j) {
    response <- x[[j]]
    predictors <- x[-j]
    fit <- stats::lm(response ~ ., data = predictors)
    r2 <- summary(fit)$r.squared
    tolerance <- max(1 - r2, .Machine$double.eps)
    data.frame(
      variable = names(x)[j],
      vif = 1 / tolerance,
      tolerance = tolerance
    )
  })
  dplyr::bind_rows(rows)
}

diagnose_collinearity <- function(dat, cfg, hazard, output_dir) {
  design <- collinearity_design(dat, cfg)
  x <- design$x
  scaled <- scale(x)
  correlation <- stats::cor(x)
  vif <- compute_vif(x)
  condition_number <- tryCatch(
    kappa(scaled, exact = TRUE),
    error = function(e) Inf
  )

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  corr_frame <- data.frame(variable = rownames(correlation), correlation,
                           row.names = NULL, check.names = FALSE)
  utils::write.csv(
    corr_frame,
    file.path(output_dir, paste0(hazard, "_covariate_correlation.csv")),
    row.names = FALSE
  )
  utils::write.csv(
    vif,
    file.path(output_dir, paste0(hazard, "_covariate_vif.csv")),
    row.names = FALSE
  )
  condition <- data.frame(
    hazard = hazard,
    n_event_ring_rows = design$n_complete,
    n_covariates = ncol(x),
    condition_number = condition_number,
    max_vif = max(vif$vif),
    regularizing_prior_sd = COVARIATE_PRIOR_SD
  )
  utils::write.csv(
    condition,
    file.path(output_dir, paste0(hazard, "_condition_number.csv")),
    row.names = FALSE
  )
  list(correlation = correlation, vif = vif, condition = condition)
}
