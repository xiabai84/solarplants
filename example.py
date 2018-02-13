import sp_googlemaps
import importlib

if __name__ == "__main__":
    # for development, reload the package every time
    importlib.reload(sp_googlemaps)

    # load google api key:
    api_key = open('googleapikey.txt', 'r').readline().strip()

    file = open('doc/Locations_Matthias.csv', 'r')
    #file = open('doc/Locations_Bavaria.csv', 'r')
    # Get rid of header:
    _ = file.readline()

    # To resume an aborted download run, enter number of images to skip
    skip_first = 236
    for line in file.readlines():
        (postal_code, city, street) = line.strip().split(';')
        for house_number in range(1,11):
            if skip_first > 0:
                skip_first -= 1
                continue
            address = "{} {}, {} {}".format(street, house_number, postal_code, city)
            sp_googlemaps.download_satellite_image(address, 'images', key=api_key)

    file.close()
