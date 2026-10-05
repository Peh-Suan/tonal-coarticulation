###############################################################################
#  Matched-retention robustness check for Experiment 1
#
#  Reviewer 2 asked whether the production results hold "under an analysis that
#  accounts for this imbalance" in discard rates across tone categories.
#
#  Approach: within each language group, randomly discard RETAINED tokens so
#  that every tone combination contributes the same PROPORTION of its design
#  total (10 repetitions x n speakers). Every tone pair then enters the model at
#  an identical retention rate, so the compositional imbalance is removed.
#  Repeat over many random draws and check the coarticulation slopes and the
#  cross-linguistic contrasts.
#
#  NOTE ON INTERPRETATION: this removes imbalance in how much each tone
#  combination contributes. It does NOT undo the selection itself — the lost
#  tokens are the low-F0 ones and cannot be recovered by any reweighting. That
#  part is addressed by the direction-of-bias argument (selection on the
#  dependent variable attenuates slopes toward zero).
###############################################################################

library(lme4); library(lmerTest); library(emmeans); library(dplyr)

N_ITER <- 20              # start at 3 to time it, then raise
MATCH  <- "within_lang"   # "within_lang" (default) or "global"
SEED   <- 2026

## ---- 1. data, exactly as in data_analysis_filtered.ipynb -------------------
df <- read.csv("concatenated_data/f0_z_onset_offset_processed.csv")
df$tone                <- as.factor(df$tone)
df$lang                <- as.factor(df$lang)
df$onset_segment_type  <- as.factor(df$onset_segment)
df$offset_segment_type <- as.factor(df$offset_segment)
df$position            <- as.factor(df$position)
df$participant         <- as.factor(df$participant)
df <- df[!df$participant %in% c("P9", "P11"), ]
df$onset_portion_idx  <- scale(df$onset_portion_idx)
df$offset_portion_idx <- scale(df$offset_portion_idx)

FORM <- y ~ x*lang*tone*position*onset_portion_idx*offset_portion_idx +
            (x | onset_segment_type) + (x | offset_segment_type) + (x | participant)
CTRL <- lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)

## ---- 2. reproduce the published analysis set (2.5 SD residual trim) --------
m0 <- lmer(FORM, df, control = CTRL)
df$model_resid <- resid(m0)
df_clean <- df %>% filter(as.vector(abs(scale(model_resid))) < 2.5)
cat("\nanalysis rows:", nrow(df_clean),
    "| participants:", nlevels(droplevels(df_clean$participant)), "\n")

# AT_MODE "max"  = replicate data_analysis_filtered.ipynb (slope at the MAXIMUM
#                  portion index, i.e. the most interior measurement point)
# AT_MODE "mean" = slope at the mean portion index, which is what the main
#                  effect of x in Table 2 corresponds to
AT_MODE <- "mean"
AT <- if (AT_MODE == "max") {
  list(onset_portion_idx  = max(df_clean$onset_portion_idx),
       offset_portion_idx = max(df_clean$offset_portion_idx))
} else {
  list(onset_portion_idx = 0, offset_portion_idx = 0)
}
cat("\nemtrends evaluated at portion-index mode:", AT_MODE, "\n")

get_trends <- function(model) {
  tr <- emtrends(model, ~ lang, var = "x", weights = "flat", at = AT)
  list(slopes    = as.data.frame(summary(tr, infer = c(TRUE, TRUE))),
       contrasts = as.data.frame(pairs(tr)))
}

## ---- 3. retention per language x tone combination -------------------------
df_clean$tokid <- paste(df_clean$participant, df_clean$lang,
                        df_clean$pair, df_clean$token, sep = "|")
tok <- df_clean %>% distinct(participant, lang, pair, token, tokid)

nspk <- tok %>% group_by(lang) %>% summarise(n = n_distinct(participant), .groups = "drop")
ret  <- tok %>% count(lang, pair, name = "kept") %>%
  left_join(nspk, by = "lang") %>%
  mutate(total = 10 * n, retention = kept / total)

if (MATCH == "global") {
  ret$target <- min(ret$retention)
} else {
  ret <- ret %>% group_by(lang) %>% mutate(target = min(retention)) %>% ungroup()
}
ret$keep_n <- pmax(1, floor(ret$target * ret$total))

cat("\n--- retention by language x tone pair ---\n")
print(as.data.frame(ret %>% arrange(lang, retention)), row.names = FALSE)
cat("\nmatched target retention per language:\n")
print(as.data.frame(ret %>% group_by(lang) %>%
        summarise(target = 100*first(target), tokens_kept = sum(keep_n), .groups="drop")),
      row.names = FALSE)

## ---- 4. original model, for reference --------------------------------------
cat("\n===================== ORIGINAL (UNMATCHED) MODEL =====================\n")
g0 <- get_trends(lmer(FORM, df_clean, control = CTRL))
print(g0$slopes, row.names = FALSE); cat("\n"); print(g0$contrasts, row.names = FALSE)

## ---- 5. matched subsets ----------------------------------------------------
set.seed(SEED)
res_s <- list(); res_c <- list(); t0 <- Sys.time()
for (i in seq_len(N_ITER)) {
  sel <- as.data.frame(
    tok %>% left_join(ret %>% select(lang, pair, keep_n), by = c("lang", "pair")))
  sel <- do.call(rbind, lapply(
    split(sel, list(sel$lang, sel$pair), drop = TRUE),
    function(g) g[sample(nrow(g), min(g$keep_n[1], nrow(g))), , drop = FALSE]))
  sub <- df_clean[df_clean$tokid %in% sel$tokid, ]
  sub$tone <- droplevels(sub$tone); sub$lang <- droplevels(sub$lang)

  ok <- tryCatch({
    m <- lmer(FORM, sub, control = CTRL); g <- get_trends(m)
    res_s[[length(res_s)+1]] <- cbind(iter = i, g$slopes)
    res_c[[length(res_c)+1]] <- cbind(iter = i, g$contrasts); TRUE
  }, error = function(e) { cat("  iteration", i, "failed:", conditionMessage(e), "\n"); FALSE })
  cat("iteration", i, "/", N_ITER, "- rows", nrow(sub),
      "-", ifelse(ok, "ok", "SKIPPED"),
      "-", round(difftime(Sys.time(), t0, units = "mins"), 1), "min elapsed\n")
}

## ---- 6. summary ------------------------------------------------------------
S <- bind_rows(res_s); C <- bind_rows(res_c)
cat("\n============ MATCHED SUBSETS (", length(unique(S$iter)), "draws ) ============\n")
cat("\nby-language coarticulation slope (x.trend):\n")
print(as.data.frame(S %>% group_by(lang) %>% summarise(
        mean = mean(x.trend), sd = sd(x.trend),
        min = min(x.trend), max = max(x.trend),
        pct_significant = 100*mean(p.value < .05), .groups = "drop")), row.names = FALSE)
cat("\ncross-linguistic contrasts:\n")
print(as.data.frame(C %>% group_by(contrast) %>% summarise(
        mean = mean(estimate), sd = sd(estimate),
        min = min(estimate), max = max(estimate),
        pct_significant = 100*mean(p.value < .05), .groups = "drop")), row.names = FALSE)

write.csv(S, "matched_retention_slopes_atmean.csv",    row.names = FALSE)
write.csv(C, "matched_retention_contrasts_atmean.csv", row.names = FALSE)
# cat("\nwritten: matched_retention_slopes_atmean.csv, matched_retention_contrasts_atmean.csv\n")
