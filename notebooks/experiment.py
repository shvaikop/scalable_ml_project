# from smhi_open_data import SMHIOpenDataClient, Parameter
#
#
# # Get 10 stations
# client = SMHIOpenDataClient()
#
# # Get all stations
# # stations = client.get_stations()
# #
# # print(stations)
#
# # Get closest station
# closest_station = client.get_closest_station(
#     latitude=59.3284,
#     longitude=18.0664)
#
# print(closest_station)
#
# # Get available parameters
# # parameters = client.list_parameters()
# #
# # # Get available parameters at station
# # parameters_station = client.get_station_parameters(station_id=173010)
# #
# # # Get temperature observations from available stations from past hour
# # observations = client.get_latest_observations(
# #     parameter=Parameter.TemperaturePast1h)


from datetime import datetime, timezone
from smhi_open_data import SMHIOpenDataClient, Parameter

client = SMHIOpenDataClient()

station_id = 20389  # Mälaren W
date = datetime(2024, 1, 15, tzinfo=timezone.utc)

observations = client.get_observations(
    station_id=station_id,
    parameter=Parameter.WaterLevel,
    start_time=date,
    end_time=date
)

for obs in observations:
    print(obs.date, obs.value)

