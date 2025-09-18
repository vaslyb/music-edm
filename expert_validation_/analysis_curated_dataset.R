# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(irr)
library(corrplot)

# Set working directory
setwd("C:/Users/au757988/OneDrive - Aarhus universitet/Desktop/PhD MIB/3.MIR/Expert_Validation/Analysis/3.analysis_curated_dataset")

# Load the datasets
selected_songs <- read.csv("top_50_songs_per_genre.csv")
all_experts <- read.csv("all_experts_zscores.csv")

# Merge datasets to get expert ratings for the selected songs
selected_expert_ratings <- all_experts %>%
  inner_join(selected_songs, by = c("audioFile", "genre"))

# 1. DESCRIPTIVE STATISTICS FOR THE SELECTED SONGS (200 SONGS)
# Overall descriptive statistics (mean, sd, min, max)
selected_stats <- selected_expert_ratings %>%
  summarise(mean_z = mean(z_score),
            sd_z = sd(z_score),
            min_z = min(z_score),
            max_z = max(z_score))

write.csv(selected_stats, "selected_songs_overall_stats.csv", row.names = FALSE)

# Descriptive statistics by genre (mean, sd, min, max)
genre_stats_selected <- selected_expert_ratings %>%
  group_by(genre) %>%
  summarise(mean_z = mean(z_score),
            sd_z = sd(z_score),
            min_z = min(z_score),
            max_z = max(z_score),
            num_songs = n())

write.csv(genre_stats_selected, "selected_songs_genre_stats.csv", row.names = FALSE)

# 2. AGREEMENT STATISTICS (SD, MAD)
agreement_stats_selected <- selected_expert_ratings %>%
  group_by(audioFile) %>%
  summarise(sd_z = sd(z_score),
            mad_z = mad(z_score))

write.csv(agreement_stats_selected, "selected_songs_agreement_stats.csv", row.names = FALSE)

# 3. OUTLIER ANALYSIS
outliers_selected <- selected_expert_ratings %>%
  filter(abs(z_score) > 2)

print(paste("Number of Outliers in the Selected Songs:", nrow(outliers_selected)))
write.csv(outliers_selected, "selected_songs_outliers.csv", row.names = FALSE)

# Report the number of outliers per genre
outlier_stats_selected <- outliers_selected %>%
  group_by(genre) %>%
  summarise(num_outliers = n())

write.csv(outlier_stats_selected, "selected_songs_outlier_stats.csv", row.names = FALSE)

# 4. CORRELATION ANALYSIS AMONG EXPERTS (INCLUDING PER GENRE)
# Prepare data for correlation analysis (expert-wise ratings)
expert_correlation_selected <- selected_expert_ratings %>%
  select(audioFile, Expert.ID, z_score) %>%
  pivot_wider(names_from = Expert.ID, values_from = z_score)

# Compute correlation matrix for all selected songs
cor_matrix_selected <- cor(expert_correlation_selected[, -1], use = "complete.obs")

# Save the correlation matrix
write.csv(cor_matrix_selected, "correlation_matrix_selected_songs.csv", row.names = TRUE)

# Save overall correlation matrix plot (white background)
png("selected_songs_correlation_matrix_plot.png", width = 800, height = 800)
corrplot(cor_matrix_selected, method = "color", bg = "white")  
dev.off()

# Correlation analysis and plots for each genre
genres <- unique(selected_expert_ratings$genre)

for (g in genres) {
  # Filter data by genre
  genre_data <- selected_expert_ratings %>%
    filter(genre == g) %>%
    select(audioFile, Expert.ID, z_score) %>%
    pivot_wider(names_from = Expert.ID, values_from = z_score)
  
  # Compute correlation matrix for the genre
  cor_matrix <- cor(genre_data[, -1], use = "complete.obs")
  
  # Save correlation matrix to CSV
  write.csv(cor_matrix, paste0("correlation_matrix_", g, ".csv"), row.names = TRUE)
  
  # Save correlation plot for the genre
  png(paste0("correlation_matrix_plot_", g, ".png"), width = 800, height = 800)
  
  # Adjust margins for the title
  par(mar=c(5, 4, 8, 2) + 0.1)  # Increase top margin for the title
  
  # Generate the correlation plot
  corrplot(cor_matrix, method = "color", bg = "white", tl.col = "black", tl.srt = 45)
  
  # Add title using mtext for more control
  mtext(text = paste("Correlation Matrix -", g), side = 3, line = 6, cex = 2)  # Top side, line 6 for spacing
  
  dev.off()
}

# 5. INTER-RATER RELIABILITY (ICC) FOR ALL SELECTED SONGS
# Compute ICC for the whole dataset
icc_result_selected <- icc(expert_correlation_selected[, -1], model = "twoway", type = "consistency", unit = "average")

# Create a data frame to store ICC results for the whole dataset
icc_results_df <- data.frame(Genre = "All",
                             ICC_Value = icc_result_selected$value,
                             Lower_Bound_CI = icc_result_selected$lbound,
                             Upper_Bound_CI = icc_result_selected$ubound,
                             P_Value = icc_result_selected$p.value)

# Loop over each genre to compute ICC and save the results
for (g in genres) {
  # Filter data by genre
  genre_data <- selected_expert_ratings %>%
    filter(genre == g) %>%
    select(audioFile, Expert.ID, z_score) %>%
    pivot_wider(names_from = Expert.ID, values_from = z_score)
  
  # Check if there are enough experts to compute ICC
  if (ncol(genre_data) > 1) {  # At least one audioFile and one Expert.ID
    # Compute ICC for the genre
    icc_result <- icc(genre_data[, -1], model = "twoway", type = "consistency", unit = "average")
    
    # Create a temporary data frame to store ICC results for the genre
    icc_result_df <- data.frame(Genre = g,
                                ICC_Value = icc_result$value,
                                Lower_Bound_CI = icc_result$lbound,
                                Upper_Bound_CI = icc_result$ubound,
                                P_Value = icc_result$p.value)
    
    # Append the genre's ICC result to the main data frame
    icc_results_df <- bind_rows(icc_results_df, icc_result_df)
  }
}

# Save the ICC results table for both the whole dataset and per genre
write.csv(icc_results_df, "icc_results_all_and_by_genre.csv", row.names = FALSE)

# Print message
cat("ICC results for the whole dataset and per genre have been saved as icc_results_all_and_by_genre.csv.\n")

# 6. VISUALIZATIONS
# Boxplots of z-scores by genre
boxplot_genre <- ggplot(selected_expert_ratings, aes(x = genre, y = z_score)) +
  geom_boxplot() +
  ggtitle("Z-scores of Selected Songs by Genre")
 
  ggsave("selected_songs_boxplot_genre.png", plot = boxplot_genre, width = 8, height = 6)

# Density plot for z-scores
density_plot <- ggplot(selected_expert_ratings, aes(x = z_score)) +
  geom_density(fill = "blue", alpha = 0.4) +
  ggtitle("Density of Z-scores for Selected Songs")

ggsave("selected_songs_density_plot.png", plot = density_plot, width = 8, height = 6)

# Scatter plot of mean z-scores vs agreement (SD), colored by genre
agreement_plot_data_selected <- selected_expert_ratings %>%
  group_by(audioFile, genre) %>%
  summarise(mean_z = mean(z_score), sd_z = sd(z_score))

# Define custom colors for the genres
genre_colors <- c("progressive house" = "pink",
                  "minimal techno" = "black",
                  "psytrance" = "green",
                  "dubstep" = "blue")

agreement_plot_selected <- ggplot(agreement_plot_data_selected, aes(x = mean_z, y = sd_z, color = genre)) +
  geom_point() +
  scale_color_manual(values = genre_colors) +
  ggtitle("Mean Z-scores vs. Agreement (SD) for Selected Songs by Genre")
 
ggsave("selected_songs_mean_vs_agreement_colored_plot.png", plot = agreement_plot_selected, width = 8, height = 6)
