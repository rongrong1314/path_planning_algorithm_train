#you can use this code to convert GPS date(longitude latitude altitude)
# to local(north east down) coordinates(GEO)
import utm
import numpy as np

#first you should define a global home position as the origin of you global_home
def global_to_local(global_position,global_home):
    #convert the coordinates so that the latitude and longitude are converted to x,y
    (east_home,north_home,_,_) = utm.from_latlon(global_home[1],global_home[0])
    (east,north,_,_) = utm.from_latlon(global_position[1],global_position[0])
    local_position = np.array([north-north_home,east-east_home, -(global_position[2]-global_position[2])])
    return local_position
#NED to Geodetic
def local_to_global(local_position,global_home):
    (east_home,north_home,zone_number,zone_letter) = utm.from_latlon(global_home[1],global_home[0])
    (lat,lon) = utm.to_latlon(east_home+local_position[1],north_home+local_position[0],zone_number,zone_letter)
    global_position = np.array([lon,lat,-(local_position[2]-global_home[2])])
    return global_position

#set output format
np.set_printoptions(precision=2)

geodetic_current_coordinates = [-122.079465,37.393037,30]
geodetic_home_coordinates = [-122.108432,37.400154,20]
local_coordinates_NED = global_to_local(geodetic_current_coordinates,geodetic_home_coordinates)
print(local_coordinates_NED)

np.set_printoptions(precision=6)
NED_coordinates = [25.21,128.07,-30.]
geodetic_current_coordinates = local_to_global(NED_coordinates,geodetic_home_coordinates)
print(geodetic_current_coordinates)