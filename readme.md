# San Francisco bike sharing system
## by Javier Alonso Alonso


## Dataset

> My main dataframe (df_trips) contains all the trips made on the bike sharing system of San Francisco and after all the wrangling issues have the next fields:

> - start_time: when the trip began
> - start_station_id: the identifier of the station where the trip begins
> - end_station_id: the identifier of the station where the trip ends
> - bike_id: the identifier of the bike used on the trip
> - user_type: if the user of the bike is a Subscriber or a Customer
> - duration_min: the duration of the trip in minutes
> - date: the day, month and year when the trip begins
> - month: the month where the trip begins
> - day_week: the day of the week when the trip begins, being 1 Monday, 2 Tuesday,... until 7 Sunday

> There´s also an auxiliary dataframe (df_stations) that contain information of the bike stations, having the next fields:

> - id: identifier of the station, related to the id´s of the start and end station of the main dataframe
> - name: name of the station
> - latitude: latitude of the station
> - longitude: longitude of the station

> To get this final dataframes I´ve made many wrangling issues (gathering, assesing and cleaning) that are explained in detail in "exploration_template.ipynb"

> Once made the wrangling issues the observatory actions were made, begining with univariate analysis to multivariate in the next step.

## Summary of Findings

> There are two types of users, subscribers and customers. Subscribers are the users that use the bike on a daily basis, most of them San Francisco citizens, and customers are sporadic users, most of them tourists, and both types behave differently using the bikesharing system.

> The duration of the tourists trips are much longer than the ones of the citizens because they use the bike for tourism and for travelling around the city, while the subscribers use it for going to one place to other directly, and during the non weekend days the majority of the trips are from home to job or viceversa. In both cases the duration of the the trips are quite longer during weekends.

> The number of trips taken by subscribers and costumers vary a lot, beeing much higher the trips of the citizens along the year. The number of trips for customers is almost constant during the seven years of the week because the number of tourists don´t vary too much between days. On the other hand the trips of subscribers descend a lot during weekends because most of the trips out of the weekend are due to the transport between job and home and viceversa

> During the two years per month the subscribers trips are always higher than the costumers but both behave in a similar way, being greater in the middle months of the year and being lower in winter, except from last month analyzed, December 2019, were both have similar number of trips due to a high descend in citizens trips and a high uprise in tourist trips. Probably there´s a reason for this abnomal behaviour.


## Key Insights for Presentation

> Select one or two main threads from your exploration to polish up for your presentation. Note any changes in design from your exploration step here.

> I´m going to polisth up for my presentation the two next threads:

> - Number of trips vs Month vs User
> - Number of trips vs Weekday vs User
> - Duration vs Weekday vs User