library(readr)
library(tidyverse)
library(scales)
library(plotly)
Full_centrality <- read_csv("C:/Users/Jackyfirst/Downloads/BDSRC/research/Full_centrality_without_politicians.csv", 
                            col_types = cols(X1 = col_skip(), 
                                             page_id = col_character(), 
                                             page_name = col_character(),
                                             week = col_date(format = "%Y-%m-%d")))

page_name_map = read_csv("C:/Users/Jackyfirst/Downloads/BDSRC/research/1000-page-info.csv",
                         col_types = cols( 
                           page_id = col_character(), 
                           page_name = col_character()
                         )
)[,1:2]


Full_centrality = Full_centrality %>% left_join(page_name_map, by='page_id') %>%relocate(page_name, .after = page_id) 

get_top_20 = function(centrality){
  Full_centrality %>% 
    filter(week == min(week)) %>% 
    arrange(desc(.data[[centrality]])) %>% 
    select(page_id) %>% head(127) %>% pull
}

plot_top_20 = function(top_20_list, centrality){
  top20.centrality.all = Full_centrality %>% filter(page_id %in% top_20_list) %>% select(page_name,.data[[centrality]], week)
  
  (top20.centrality.all %>% ggplot() + 
      geom_line(aes(y = .data[[centrality]], x = week, color=page_name)) +
      scale_x_date(labels = date_format("%Y-%m"))) %>% ggplotly
}

centrality = 'eigenvector_centrality'
top.20.eig = get_top_20(centrality)
plot_top_20(top.20.eig, centrality)