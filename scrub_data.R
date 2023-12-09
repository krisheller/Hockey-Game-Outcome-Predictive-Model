#Credit: Christian Lee
#Source: https://medium.com/hockey-stats/how-to-scrape-nhl-com-dynamic-data-in-r-using-rvest-and-rselenium-ba3b5d87c728

library(RSelenium)
library(rvest)
#library(dplyr)
library(wdman)
selServ <- wdman::selenium(retcommand = TRUE, verbose = FALSE)

url = "http://www.nhl.com/stats/teams?aggregate=0&report=summaryshooting&reportType=game&dateFrom=2018-10-02&dateTo=2019-04-06&gameType=2&homeRoad=H&filter=gamesPlayed,gte,1&sort=satTotal&page=0&pageSize=10"
rD = rsDriver(port=4447L, browser="chrome", chromever="114.0.5735.16") #specify chrome version
remDr = rD[['client']]
remDr$open()
remDr$navigate(url) #this will open a chrome window
src = remDr$getPageSource()[[1]] #select everything for now

df = read_html(src) %>% 
 xml_nodes(xpath='//*[contains(concat( " ", @class, " " ), concat( " ", "rt-td”, " " ))]') %>%
 xml_text() %>%
 matrix(.,ncol=19, byrow = T) %>%
 data.frame(.,stringsAsFactors = F)