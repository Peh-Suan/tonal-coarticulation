## -----------------------------------------------------------------------------
library(lme4)
library(lmerTest)
library(car)
library(ggplot2)


## -----------------------------------------------------------------------------
df <- read.csv("concatenated_data/experiment2_results.csv")
df$lang <- as.factor(df$lang)
df$preceding_tone <- as.factor(df$preceding_tone)
df$participant <- as.factor(df$participant)
df$word_freq_ratio <- scale(df$word_freq_ratio)

df$preceding_tone_offset <- as.integer(sub(".*(\\d)$", "\\1", df$preceding_tone))

cor(df$word_freq_ratio, df$falling_tone_chosen, method = "pearson")
cor(df$word_freq_ratio, df$preceding_tone_offset, method = "pearson")


## -----------------------------------------------------------------------------
temp <- df

contrasts(temp$preceding_tone) <- contr.sum(length(unique(temp$preceding_tone)))
print(contrasts(temp$preceding_tone))
model <- glmer(falling_tone_chosen ~ level*lang + preceding_tone_offset*lang + preceding_tone*lang + (level|participant), family=binomial, data=temp, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
summary(model)


## -----------------------------------------------------------------------------
temp$residual <- residuals(model)
model <- lmer(residual ~ word_freq_ratio + (1|participant), data=temp)
summary(model) # Post-hoc residual analysis to see if word frequency ratio is a significant predictor of the remaining variance


## -----------------------------------------------------------------------------
temp <- df
temp <- temp[(temp$lang =="min") | (temp$lang =="man_bilingual"),]
temp$lang <- droplevels(temp$lang)
contrasts(temp$preceding_tone) <- contr.sum(length(unique(temp$preceding_tone)))
print(contrasts(temp$lang))
print(contrasts(temp$preceding_tone))
model <- glmer(falling_tone_chosen ~ level*lang + preceding_tone_offset*lang + preceding_tone*lang + (level|participant), family=binomial, data=temp, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
summary(model)


## -----------------------------------------------------------------------------
# Discussion on bilingualism: Does fluency in TSM affect perceptual compensation in TM?
temp <- df
temp <- temp[(temp$lang =="man_bilingual"),]
temp$preceding_tone <- droplevels(temp$preceding_tone)
temp$lang <- droplevels(temp$lang)
contrasts(temp$preceding_tone) <- contr.sum(length(unique(temp$preceding_tone)))
print(contrasts(temp$preceding_tone))
model <- glmer(falling_tone_chosen ~ level*fluency + preceding_tone_offset*fluency + preceding_tone*fluency + (level|participant), family=binomial, data=temp, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
summary(model)

