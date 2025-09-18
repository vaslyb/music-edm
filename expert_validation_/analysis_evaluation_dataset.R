# Load required libraries
library(dplyr)
library(tidyr)
library(irr)
library(corrplot)
library(ggplot2)

# Load the merged z-score dataset
setwd("C:/Users/au757988/OneDrive - Aarhus universitet/Desktop/PhD MIB/3.MIR/Expert_Validation/Analysis/2.analysis_400_audio")
all_experts <- read.csv("all_experts_zscores.csv")

# 1. INTER-RATER RELIABILITY
# Calculate SD and MAD for each song
agreement_stats <- all_experts %>%
  group_by(audioFile) %>%
  summarise(sd_z = sd(z_score),
            mad_z = mad(z_score))

# Print and save summary of agreement statistics
print(agreement_stats)
write.csv(agreement_stats, "agreement_stats.csv", row.names = FALSE)

# 2. Format the data for ICC: each row is a song, each column is an expert's rating
icc_data <- all_experts %>%
  select(audioFile, Expert.ID, z_score) %>%
  pivot_wider(names_from = Expert.ID, values_from = z_score)

# Check the reshaped data
print(head(icc_data))

# Ensure that there are no NA values in the ICC data
icc_data <- na.omit(icc_data)

# Compute ICC for the entire dataset
icc_result_all <- icc(icc_data[, -1], model = "twoway", type = "consistency", unit = "average")

# Format ICC result for the overall dataset
icc_results_df <- data.frame(
  Genre = "All",
  ICC_Value = icc_result_all$value,
  Lower_Bound_CI = icc_result_all$lbound,
  Upper_Bound_CI = icc_result_all$ubound,
  P_Value = icc_result_all$p.value,
  stringsAsFactors = FALSE
)

# 3. Calculate ICC for each genre and store results
genres <- unique(all_experts$genre)

for (g in genres) {
  # Filter data for the current genre
  genre_data <- all_experts %>%
    filter(genre == g) %>%
    select(audioFile, Expert.ID, z_score) %>%
    pivot_wider(names_from = Expert.ID, values_from = z_score)
 
    # Compute ICC for the genre
  icc_result_genre <- icc(genre_data[, -1], model = "twoway", type = "consistency", unit = "average")
    
   # Append the genre's ICC result to the results data frame
  icc_results_df <- rbind(icc_results_df, 
                            data.frame(Genre = g,
                                       ICC_Value = icc_result_genre$value,
                                       Lower_Bound_CI = icc_result_genre$lbound,
                                       Upper_Bound_CI = icc_result_genre$ubound,
                                       P_Value = icc_result_genre$p.value))
  }

# Save the ICC results table for both the whole dataset and per genre
write.csv(icc_results_df, "icc_results_all_and_by_genre.csv", row.names = FALSE)


#SELECT THE 50 MOST AGREED-UPON SONGS PER GENRE
# Calculate mean z-score for each song
mean_z_scores <- all_experts %>%
  group_by(audioFile, genre) %>%
  summarise(mean_z = mean(z_score))

# Rank songs by genre based on their mean z-scores
ranked_songs <- mean_z_scores %>%
  arrange(genre, desc(mean_z)) %>%
  group_by(genre) %>%
  slice(1:50)  # Select top 50 songs per genre

# Save the top 50 songs per genre
write.csv(ranked_songs, "top_50_songs_per_genre.csv", row.names = FALSE)

#EXTRA ANALYSES
# 1. Identify outliers based on z-scores 
# Set threshold for outliers
outlier_threshold <- 2

outliers <- all_experts %>%
  group_by(audioFile, genre) %>%
  filter(abs(z_score) > outlier_threshold)

# Report the number of outliers per genre
outlier_stats <- outliers %>%
  group_by(genre) %>%
  summarise(num_outliers = n())

print(outlier_stats)
# Save outlier statistics per genre
write.csv(outlier_stats, "outlier_stats.csv", row.names = FALSE)

# Save the list of outlier songs
write.csv(outliers, "outlier_songs.csv", row.names = FALSE)


# 2. Descriptive statistics for the full dataset (z-scores)
full_dataset_stats <- all_experts %>%
  summarise(mean_z = mean(z_score),
            sd_z = sd(z_score),
            min_z = min(z_score),
            max_z = max(z_score))

print(full_dataset_stats)
write.csv(full_dataset_stats, "full_dataset_stats.csv", row.names = FALSE)

# Descriptive statistics by genre
genre_stats <- all_experts %>%
  group_by(genre) %>%
  summarise(mean_z = mean(z_score),
            sd_z = sd(z_score),
            min_z = min(z_score),
            max_z = max(z_score))

print(genre_stats)
write.csv(genre_stats, "genre_stats.csv", row.names = FALSE)

#3. Create a correlation matrix of experts' ratings
expert_correlation <- all_experts %>%
  select(Expert.ID, audioFile, z_score) %>%
  spread(key = Expert.ID, value = z_score)

cor_matrix <- cor(expert_correlation[, -1], use = "complete.obs")
write.csv(cor_matrix, "expert_correlation_matrix.csv", row.names = TRUE)

# Save the correlation matrix as an image
png("correlation_matrix_plot.png", width = 800, height = 800)
corrplot(cor_matrix, method = "color")
dev.off()

# Correlation analysis and plots for each genre
genres <- unique(all_experts$genre)

for (g in genres) {
  # Filter data by genre
  genre_data <- all_experts %>%
    filter(genre == g) %>%
    select(audioFile, Expert.ID, z_score) %>%
    pivot_wider(names_from = Expert.ID, values_from = z_score)
  
  # Compute correlation matrix for the genre
  cor_matrix_genre <- cor(genre_data[, -1], use = "complete.obs")
  
  # Save correlation matrix to CSV
  write.csv(cor_matrix_genre, paste0("correlation_matrix_", g, ".csv"), row.names = TRUE)
  
  # Save correlation plot for the genre
  png(paste0("correlation_matrix_plot_", g, ".png"), width = 800, height = 800)
  
  # Adjust margins for the title
  par(mar=c(5, 4, 8, 2) + 0.1)  # Increase top margin for the title
  
  # Generate the correlation plot
  corrplot(cor_matrix_genre, method = "color", bg = "white", tl.col = "black", tl.srt = 45)
  
  # Add title
  mtext(text = paste("Correlation Matrix -", g), side = 3, line = 6, cex = 2)
  
  dev.off()
}

# 4.Visualization of normalization 
# Density plot for pre-normalization ratings
pre_norm_plot <- ggplot(all_experts, aes(x = slider.response)) + 
  geom_density(fill = "blue", alpha = 0.4) + 
  ggtitle("Pre-normalization Rating Distribution")

# Save the pre-normalization density plot
ggsave(filename = "pre_normalization_density_plot.png", plot = pre_norm_plot, width = 8, height = 6)

# Density plot for post-normalization (z-scores)
post_norm_plot <- ggplot(all_experts, aes(x = z_score)) + 
  geom_density(fill = "green", alpha = 0.4) + 
  ggtitle("Post-normalization Z-score Distribution")

# Save the post-normalization density plot
ggsave(filename = "post_normalization_density_plot.png", plot = post_norm_plot, width = 8, height = 6)


# 5. Scatter plot of mean z-scores vs agreement (SD), including genre information
agreement_plot_data <- all_experts %>%
  group_by(audioFile, genre) %>%
  summarise(mean_z = mean(z_score), sd_z = sd(z_score))

# Define the colors for each genre
genre_colors <- c("minimal techno" = "black", 
                  "psytrance" = "green", 
                  "dubstep" = "blue", 
                  "progressive house" = "pink")

# Create the plot with color coding by genre
agreement_plot <- ggplot(agreement_plot_data, aes(x = mean_z, y = sd_z, color = genre)) +
  geom_point() +
  scale_color_manual(values = genre_colors) +   # Manually set colors for genres
  ggtitle("Mean Z-scores vs. Agreement (SD)")

# Save the plot
ggsave("mean_z_vs_agreement_sd_colored.png", plot = agreement_plot, width = 8, height = 6)

# Boxplots of z-scores by genre
boxplot_plot <- ggplot(all_experts, aes(x = genre, y = z_score)) +
  geom_boxplot() +
  ggtitle("Z-scores by Genre")
ggsave("z_scores_by_genre_boxplot.png", width = 8, height = 6)
