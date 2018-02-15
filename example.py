import sp_googlemaps
import importlib

if __name__ == "__main__":
    # for development, reload the package every time
    importlib.reload(sp_googlemaps)

    # load google api key:
    api_key = open('googleapikey.txt', 'r').readline().strip()

    #file = open('doc/Locations_L.csv', 'r')
    file = open('doc/Locations_Matthias.csv', 'r')
    #file = open('doc/Locations_Bavaria.csv', 'r')
    # Get rid of header:
    _ = file.readline()

    for line in file.readlines():
        (postal_code, city, street, starting_street_number) = line.strip().split(';')

        street_numbers_found = 0
        street_number = starting_street_number

        # TODO: Some streets have no street numbers in Google Maps. This will lead to an infinite loop.
        while street_numbers_found < 10:
            address = "{} {}, {} {}".format(street, street_number, postal_code, city)
            if sp_googlemaps.check_address_existence(address, str(street_number), api_key):
                sp_googlemaps.download_satellite_image(address, 'images', 'images/thumbs', key=api_key)
                street_numbers_found += 1
            street_number += 1

    file.close()
