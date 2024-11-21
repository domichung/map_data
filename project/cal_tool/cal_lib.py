import math

# convert pixel size into real length 
def pixel_to_ground_distance (zoom) :
    ground_distance = 156543.04 / (2 ** zoom)
    return ground_distance

# convert picture size into real length 
def calculate_real_size (width, height, zoom) :
    ground_distance = pixel_to_ground_distance(zoom)
    
    real_width_dis = width * ground_distance 
    real_height_dis = height * ground_distance 

    return [real_width_dis, real_height_dis]

# cal center lat and lng by mul of origin picture size
def new_center (center_lat, center_lng, width, height, zoom, x, y) :
    [real_pic_width, real_pic_height] = calculate_real_size(width, height, zoom)
    
    new_lng = center_lng + (real_pic_width * x / (122400 * math.cos(center_lat * math.pi / 180)))
    new_lat = center_lat - (real_pic_height * y / 122400)

    return [new_lat, new_lng]