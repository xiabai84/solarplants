import sp_googlemaps
import importlib
import datetime

if __name__ == "__main__":
    # for development, reload the package every time
    importlib.reload(sp_googlemaps)

    # load google api key:
    api_key = open('googleapikey.txt', 'r').readline().strip()

    address_file = open('doc/Locations_L.csv', 'r')
    #address_file = open('doc/Locations_Matthias.csv', 'r')
    #address_file = open('doc/Locations_Lennart.csv', 'r')
    # Get rid of header:
    _ = address_file.readline()

    now = datetime.datetime.now()
    results_file = open('{:0>4}-{:0>2}-{:0>2}_{:0>2}-{:0>2} filenames.csv'
                        .format(now.year, now.month, now.day, now.hour, now.minute),
                        'w')
    results_file.write('Filename,Solarplant\n')

    maps_downloader = sp_googlemaps.DownloadSession(api_key, 'images/test')
    for line in address_file.readlines():
        if not line.strip():
            continue
        (postal_code, city, street, starting_street_number) = line.strip().split(',')

        street_numbers_found = 0
        street_numbers_skipped = 0
        street_number = int(starting_street_number)

        while street_numbers_found < 20 and street_numbers_skipped < 20:
            address = "{} {}, {} {}".format(street, street_number, postal_code, city)
            if maps_downloader.check_address_existence(address, str(street_number)):
                filename = maps_downloader.download_satellite_image(address)
                street_numbers_found += 1

                # for duplicates, filename will be empty
                if filename:
                    results_file.write(filename+',\n')
            else:
                street_numbers_skipped += 1
            street_number += 1

    address_file.close()
    results_file.close()
