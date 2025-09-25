# Load required libraries
library(dplyr)
library(ggplot2)
library(gridExtra)

setwd("C:/Users/au757988/OneDrive - Aarhus universitet/Desktop/PhD MIB/MIR/Expert_Validation/Analysis/expert_specific_analysis")

# Initialize lists to store summary statistics
overall_stats_list <- list()
genre_stats_list <- list()

# Initialize a list to store z-score data frames
zscore_datasets_list <- list()

# Define a function to process each expert's data
process_expert_data <- function(expert_file, expert_name) {
  # Load the expert's data
  expert_data <- read.csv(expert_file)
  
  # Calculate mean, SD, and range across all songs, including mean response time (slider.rt)
  expert_stats <- expert_data %>%
    summarise(mean_rating = mean(slider.response),
              sd_rating = sd(slider.response),
              min_rating = min(slider.response),
              max_rating = max(slider.response),
              mean_rating_response = mean(slider.rt))  # New column for mean response time
  
  # Append to the overall stats list
  overall_stats_list[[expert_name]] <<- expert_stats
  
  # Calculate mean, SD, and range by genre, including mean response time (slider.rt)
  expert_genre_stats <- expert_data %>%
    group_by(genre) %>%
    summarise(mean_rating = mean(slider.response),
              sd_rating = sd(slider.response),
              min_rating = min(slider.response),
              max_rating = max(slider.response),
              mean_rating_response = mean(slider.rt))  # New column for mean response time
  
  # Append to the genre stats list
  genre_stats_list[[expert_name]] <<- expert_genre_stats
  
  # Plot histogram of overall ratings
  histogram_plot <- ggplot(expert_data, aes(x = slider.response)) +
    geom_histogram(binwidth = 5, fill = "blue", color = "black") +
    ggtitle(paste("Rating Distribution for", expert_name))
  
  # Plot boxplot of ratings by genre
  boxplot_plot <- ggplot(expert_data, aes(x = genre, y = slider.response)) +
    geom_boxplot(fill = "lightblue") +
    ggtitle(paste("Rating Distribution by Genre for", expert_name))
  
  # Combine the two plots into one image
  combined_plot <- grid.arrange(histogram_plot, boxplot_plot, ncol = 2)
  
  # Save the combined plot
  ggsave(filename = paste0(expert_name, "_combined_plot.png"), plot = combined_plot, width = 12, height = 6)
  
  # Calculate z-scores
  expert_data <- expert_data %>%
    mutate(z_score = (slider.response - mean(slider.response)) / sd(slider.response))
  
  # Save the z-score dataset
  zscore_filename <- paste0(expert_name, "_zscores.csv")
  write.csv(expert_data, zscore_filename, row.names = FALSE)
  
  # Append the z-score dataset to the list for merging later
  zscore_datasets_list[[expert_name]] <<- expert_data
}

# List of expert files and names (you can add more as needed)
expert_files <- c("expert1_merged.csv", "expert2_merged.csv","expert3_merged.csv", "expert4_merged.csv", "expert5_merged.csv","expert6_merged.csv","expert8_merged.csv")
expert_names <- c("expert 1", "expert 2","expert 3", "expert 4","expert 5","expert 6","expert 8")

# Loop through each expert and process the data
for (i in seq_along(expert_files)) {
  process_expert_data(expert_files[i], expert_names[i])
}

# Combine all overall stats into a single data frame and save it
overall_stats_df <- do.call(rbind, lapply(seq_along(overall_stats_list), function(i) {
  cbind(expert_name = expert_names[i], overall_stats_list[[i]])
}))
write.csv(overall_stats_df, "overall_stats_summary.csv", row.names = FALSE)

# Combine all genre stats into a single data frame and save it
genre_stats_df <- do.call(rbind, lapply(seq_along(genre_stats_list), function(i) {
  cbind(expert_name = expert_names[i], genre_stats_list[[i]])
}))
write.csv(genre_stats_df, "genre_stats_summary.csv", row.names = FALSE)

# Merge all z-score datasets into one and save it
all_experts_zscores <- bind_rows(zscore_datasets_list)
write.csv(all_experts_zscores, "all_experts_zscores.csv", row.names = FALSE)
