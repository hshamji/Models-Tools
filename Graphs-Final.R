#Load helpful libraries
library(quantmod) # to load data
library(ggplot2) # to plot data
library(reshape) # to manipulate dataframes (melt function makes it easier to use ggplot)
library(dplyr) # to manipulate dataframes (piping functionality for cleaner code)
library(xts) # has useful time-series operations like merge.xts and diff

#Load prices
getPrice("SPY")
getPrice("VWO")
getPrice("TMF")

# Align and truncate prices & % change data
data <- merge.xts(SPY[,4], VWO[,4], TMF[,4], all=FALSE)
dataDiffs <- na.trim(diff(data)/lag.xts(data))
                     
                     

#Create dataframes to simplify plotting. Define factor for calendar year
dataDF <- data.frame(Date=index(data), coredata(data))
dataDiffsDF <- data.frame(Date=index(dataDiffs), coredata(dataDiffs), Year=as.factor(format(index(dataDiffs), format="%Y")))

# Create plot of price processes
dataDF %>%
  melt(id="Date") %>%
  ggplot(aes(x=Date, y = value, color = variable)) +
  geom_line() + xlab(label="Date") + ylab(label="Price ($)") + 
  theme(legend.position=c(.1,.85))

#Scatterplots of daily % price changes
qplot(SPY.Close, VWO.Close, data=dataDiffsDF, color=Year) + 
  coord_cartesian(xlim = c(-.075,.075), ylim = c(-0.075,.075)) + 
  stat_smooth( aes( y = VWO.Close, x = SPY.Close), inherit.aes = FALSE, se=FALSE) +  
  theme(legend.position=c(.9,.3)) + xlab("SPY Daily % Change") +ylab("VWO Daily % Change")

qplot(SPY.Close, TMF.Close, data=dataDiffsDF, color=Year) + 
  coord_cartesian(xlim = c(-.075,.075), ylim = c(-0.075,.075)) + 
  stat_smooth( aes( y = TMF.Close, x = SPY.Close), inherit.aes = FALSE, se=FALSE ) +  
  theme(legend.position=c(.9,.3)) + xlab("SPY Daily % Change") +ylab("TMF Daily % Change")