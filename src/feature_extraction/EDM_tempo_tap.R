
library(tidyr)
library(dplyr)

# go to folder with the tap tempo analysis script
setwd("C:/Users/janst/Documents/01_Experiments/Collaborations/2024_Nasia_EDM/tempo_tap")

# read .txt file
d <- read.table("psytrance.txt", header = FALSE, sep = "", dec = ".")

# get rid of first column with tap number
d <- d[ ,2:4]

# rename columns
colnames(d) <- c("trial", "genre", "tap")

# get rid of semicolon in last column
d$tap <- gsub(";", "", d$tap)
d$tap <- as.numeric(d$tap)

# define genres
d$genre[d$genre == 1]  <- "techno"
d$genre[d$genre == 2]  <- "dubstep"
d$genre[d$genre == 3]  <- "house"
d$genre[d$genre == 4]  <- "psytrance"

# create an empty data frame to store outcomes
df <- data.frame(genre = integer(),
                 trial = integer(),
                 m_iti = double(),
                 sd_iti = double())

num_trials <- length(unique(d$trial))-1

# trial loop
for (t in 0:num_trials) {
  
  x <- d[d$trial==t, ]
  
  # compute ITIs
  x$iti = x$tap - lag(x$tap, 1)
  
  # cut NAs
  x <- na.omit(x)
  
  # how many taps before cleaning?
  x_raw <- length(x$tap)
  
  # remove outliers based on 1.5 IQR
  Q <- quantile(x$iti, probs=c(.25, .75), na.rm = FALSE)
  iqr <- IQR(x$iti)
  x <- subset(x, x$iti > (Q[1] - 1.5*iqr) & x$iti < (Q[2] + 1.5*iqr))
  
  # how many taps after cleaning?
  x_clean <- length(x$tap)

  # ITI dependent variables
  if (length(x$iti) == 0) {
    m_iti <- NA
    sd_iti <- NA
    n_iti <- NA
    iti_miss <- NA
    iti_double <- NA
    iti_clean <- NA
  } else {
    m_iti <- mean(x$iti, na.rm = TRUE)
    sd_iti <- sd(x$iti, na.rm = TRUE)
    n_iti <- length(x$iti)
    iti_clean <- x_raw - x_clean
  }
  
  # append outcomes to the data frame
  df <- rbind(df, data.frame(trial = t,
                             genre = d$genre[1],
                             m_iti = round(m_iti,2),
                             sd_iti = round(sd_iti,2),
                             n_iti = n_iti,
                             iti_clean = iti_clean))
  
}

# compute BPM from ms
df$iti_bpm <- round(60000/df$m_iti, 2)

# save outcomes as csv file
write.csv(df, paste("EDM_tempo_tap_", d$genre[1], ".csv", sep=""), row.names = FALSE)
  