## -----------------------------------------------------------------------------
library(lme4)
library(lmerTest)


## -----------------------------------------------------------------------------
df <- read.csv("../concatenated_data/f0_z_onset_offset_processed.csv")

df$tone <- as.factor(df$tone)
df$lang <- as.factor(df$lang)
df$onset_segment_type <- as.factor(df$onset_segment)
df$offset_segment_type <- as.factor(df$offset_segment)
df$position <- as.factor(df$position)
df$participant <- as.factor(df$participant)


## -----------------------------------------------------------------------------
temp <- df

temp <- temp[temp$lang == "man",]

model <- lmer(y ~ x*position*tone + (x | participant), temp)
# Add fitted values to the original data
temp$fitted <- fitted(model)

# Save the necessary columns
write.csv(temp[, c("x", "y", "fitted", "position", "tone", "participant")], "lmm_fit_man.csv", row.names = FALSE)


## -----------------------------------------------------------------------------
temp <- df

temp <- temp[temp$lang == "min",]

model <- lmer(y ~ x*position*tone + (x | participant), temp)
# Add fitted values to the original data
temp$fitted <- fitted(model)

# Save the necessary columns
write.csv(temp[, c("x", "y", "fitted", "position", "tone", "participant")], "lmm_fit_min.csv", row.names = FALSE)


## -----------------------------------------------------------------------------
temp <- df

temp <- temp[temp$lang == "man_bilingual",]

model <- lmer(y ~ x*position*tone + (x | participant), temp)
# Add fitted values to the original data
temp$fitted <- fitted(model)

# Save the necessary columns
write.csv(temp[, c("x", "y", "fitted", "position", "tone", "participant")], "lmm_fit_man_bilingual.csv", row.names = FALSE)

