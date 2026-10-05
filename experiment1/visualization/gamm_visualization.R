## -----------------------------------------------------------------------------
library(mgcv)
library(itsadug)


## -----------------------------------------------------------------------------
df <- read.csv("../concatenated_data/f0_z_mean_f0_processed_for_gamm.csv")
df <- df[!is.na(df$mean_f0),]
df$participant <- as.factor(df$participant)
df$onset_segment <- as.factor(df$onset_segment)
df$offset_segment <- as.factor(df$offset_segment)
df$tone <- as.factor(df$tone)
df$lang <- as.factor(df$lang)
df$position <- as.factor(df$position)
df$pos <- as.factor(df$pos)


## -----------------------------------------------------------------------------
# Create an empty list to store all prediction dataframes
all_preds <- list()
i <- 1  # index for storing in list

for (lang in unique(df$lang)) {
    for (pair in unique(df[df$lang == lang, ]$pair)) {
        for (position in unique(df[df$lang == lang & df$pair == pair, ]$position)) {
            this_df <- df[df$lang == lang & df$pair == pair & df$position == position,]

            if (lang=="min" & startsWith(pair, "5-")){
                this_df <- df[df$lang == lang & df$pair == pair & df$position == position & df$participant %in% c("P1",
                "P7",
                "P14",
                "P28",
                "P35",
                "P37",
                "P42"),]
                model <- bam(mean_f0 ~ s(portion_idx) + 
                            s(participant, bs="re"), data = this_df)
                    
                newdata <- data.frame(
                    portion_idx = seq(min(df$portion_idx), max(df$portion_idx), length.out = 100),
                    pair = paste(pair, "|33"),
                    lang = lang,
                    position = position,
                    participant = this_df$participant[1]
                )
                    
                pred <- predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(participant)")
                    
                newdata$fit <- pred$fit
                newdata$se <- pred$se.fit
                    
                # Store with metadata
                all_preds[[i]] <- newdata
                i <- i + 1

                this_df <- df[df$lang == lang & df$pair == pair & df$position == position & !df$participant %in% c("P1",
                "P7",
                "P14",
                "P28",
                "P35",
                "P37",
                "P42"),]
                model <- bam(mean_f0 ~ s(portion_idx) + 
                            s(participant, bs="re"), data = this_df)
                    
                newdata <- data.frame(
                    portion_idx = seq(min(df$portion_idx), max(df$portion_idx), length.out = 100),
                    pair = paste(pair, "|21"),
                    lang = lang,
                    position = position,
                    participant = this_df$participant[1]
                )
                    
                pred <- predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(participant)")
                    
                newdata$fit <- pred$fit
                newdata$se <- pred$se.fit
                    
                # Store with metadata
                all_preds[[i]] <- newdata
                i <- i + 1

                }else{
                
                model <- bam(mean_f0 ~ s(portion_idx) + 
                            s(participant, bs="re"), data = this_df)
                    
                newdata <- data.frame(
                    portion_idx = seq(min(df$portion_idx), max(df$portion_idx), length.out = 100),
                    pair = pair,
                    lang = lang,
                    position = position,
                    participant = this_df$participant[1]
                )
                    
                pred <- predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(participant)")
                
                newdata$fit <- pred$fit
                newdata$se <- pred$se.fit
                    
                # Store with metadata
                all_preds[[i]] <- newdata
                i <- i + 1
                
            }
        }
    }
}

# Combine all predictions into one big dataframe
final_preds <- do.call(rbind, all_preds)

# Write to a single CSV
write.csv(final_preds, "gamm_predictions_all.csv", row.names = FALSE)

