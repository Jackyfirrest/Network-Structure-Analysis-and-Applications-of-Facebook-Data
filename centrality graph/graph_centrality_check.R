library(readr)
library(tidyverse)
library(scales)
library(plotly)
Full_centrality <- read_csv("Full_centrality.csv", 
                            col_types = cols(X1 = col_skip(), 
                                             page_id = col_character(), 
                                             page_name = col_character(),
                                             week = col_date(format = "%Y-%m-%d")))

page_name_map = read_csv("1000-page-info.csv",
                         col_types = cols( 
                           page_id = col_character(), 
                           page_name = col_character()
                         )
)[,1:2]


Full_centrality = Full_centrality %>% left_join(page_name_map, by='page_id') %>%relocate(page_name, .after = page_id)

get_top_10 = function(centrality){
  Full_centrality %>% 
    filter(week == min(week)) %>% 
    arrange(desc(.data[[centrality]])) %>% 
    select(page_id) %>% head(10) %>% pull
}

plot_top_10 = function(top_10_list, centrality){
  top10.centrality.all = Full_centrality %>% filter(page_id %in% top_10_list) %>% select(page_name,.data[[centrality]], week)
  
  (top10.centrality.all %>% ggplot() + 
      geom_line(aes(y = .data[[centrality]], x = week, color=page_name)) +
      scale_x_date(labels = date_format("%Y-%m"))) %>% ggplotly
}
t10 = get_top_10('degree_centrality')
plot_top_10(t10, 'degree_centrality')

centrality = 'eigenvector_centrality'
top.10.eig = get_top_10(centrality)
plot_top_10(top.10.eig, centrality)

centrality = 'unweighted_eigenvector_centrality'
top.10.unw.eig = get_top_10(centrality)
plot_top_10(top.10.unw.eig, centrality)

centrality = 'closeness_centrality'
top.10.cls = get_top_10(centrality)
plot_top_10(top.10.cls, centrality)