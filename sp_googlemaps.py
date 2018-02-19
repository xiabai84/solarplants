import numpy as np
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import urllib.parse
import datetime
from io import BytesIO
import PIL
from PIL import Image
import json
import re

BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
BASE_URL_GEOCODE = 'https://maps.googleapis.com/maps/api/geocode/json?'

# Retry image download if Google throws a 500 internal error
retries = Retry(total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504])


def build_google_api_url(**kwargs):
    """

    :param \**kwargs: see below
        :param \**kwargs:
        See below

    :Keyword Arguments:
        * *center* (``string``) --
           A street address or GPS coordinates that will be passed to Google Maps

    :return: ``string`` Full URL to the Google Maps Static API call
    """
    # Default values for API call
    options = {
            'size': '300x350',
            'zoom': '20',
            'maptype': 'satellite',
            'format': 'png32', }
    options.update(kwargs)

    if 'center' not in options:
        raise TypeError("Missing required argument: 'center'")

    url = BASE_URL + urllib.parse.urlencode(options)

    return url


def check_address_existence(address, street_number_used, key=''):
    """

    :param address: Full address, which will be geocoded using Google's API
    :param street_number_used: the street number part of address
    :param key: Google API key
    :return: True if the address exists on Google Maps, False otherwise
    """
    url = BASE_URL_GEOCODE + urllib.parse.urlencode({'address': address,
                                                     'key': key})
    dl_session = requests.Session()
    dl_session.mount('https://', HTTPAdapter(max_retries=retries))
    address_req = dl_session.get(url)
    address_data = json.loads(address_req.text)

    # Google Geocode only returns the street number if it actually exists
    try:
        if address_data["status"] != "OK":
            print("Google Geocoding Error, status: " + address_data["status"])
            print("Address: " + address)
            return False

        for address_component in address_data['results'][0]['address_components']:
            if 'street_number' in address_component['types'] and address_component['short_name'] == street_number_used:
                return True
    except IndexError:
        print("IndexError for JSON processing, address " + address)
        print(address_req.text)
        raise IndexError("JSON processing failed, address " + address)
        #return False

    return False


def download_satellite_image(address, output_folder='', thumbnail_folder='', image_size=300, crop_size=25,
                             thumbnail_size=50, thumbnail_prefix='th', **kwargs):
    """

    :param address: A street address or GPS coordinates that will be passed to Google Maps.
    :param output_folder: folder path where image will be saved
    :param thumbnail_folder: folder path where thumbnails will be saved
    :param image_size: Square image of image_size x image_size pixels will be created
    :param crop_size: Use this to remove Google Copyright notices. Will be added to the
        top and bottom of the image for the Google API call, but removed before saving.
    :param thumbnail_size: if this is greater than 0, create a downsampled version of
        the output image of thumbnail_size x thumbnail_size pixels
    :param thumbnail_prefix: a string that will be prepended to each thumbnail's filename
    :param kwargs: additional arguments passed to the Google Maps API call
    :return: (str) filename of the saved image (without folder)
    """

    image_height = image_size + 2*crop_size
    image_width = image_size
    url = build_google_api_url(center=address, size='{}x{}'.format(image_width, image_height), **kwargs)

    # Use session for retries, see stackoverflow post:
    # https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
    dl_session = requests.Session()
    dl_session.mount('https://', HTTPAdapter(max_retries=retries))
    image_req = dl_session.get(url)

    if image_req.raw.status != 200:
        msg_for_403 = ''
        if image_req.raw.status == 403 and 'key' not in kwargs:
            msg_for_403 = '\nTo avoid error 403 supply your API key.'
        raise Exception(
            'HTTP Request failed with code {}, reason "{}"'
            .format(image_req.raw.status, image_req.raw.reason)
            + msg_for_403
        )

    expected_content_type = 'image/png'
    if image_req.headers['Content-Type'] != expected_content_type:
        raise Exception("Wrong content type, expecting '{}', received: '{}'"
                        .format(expected_content_type, image_req.headers['Content-Type']))

    now = datetime.datetime.now()
    image_filename = '{:0>4}-{:0>2}-{:0>2}_{:0>2}-{:0>2}-{:0>2} {}.png'\
        .format(now.year, now.month, now.day, now.hour, now.minute, now.second,
                re.sub(r'[^\w\s]+', '', address))
    print(image_filename)

    if output_folder and output_folder[-1] != '/':
        output_folder += '/'
    if thumbnail_folder and thumbnail_folder[-1] != '/':
        thumbnail_folder += '/'
    image = Image.open(BytesIO(image_req.content))
    image_box = image.getbbox()
    image = image.crop((image_box[0],
                        image_box[1]+crop_size,
                        image_box[2],
                        image_box[3]-crop_size))
    image.save(output_folder + image_filename)

    if thumbnail_size > 0:
        if (not thumbnail_folder) and (not thumbnail_prefix):
            raise ValueError('thumbnail_prefix and thumbnail_folder cannot both be empty')

        image = image.resize((thumbnail_size, thumbnail_size), resample=PIL.Image.LANCZOS)
        if not thumbnail_folder:
            thumbnail_folder = output_folder

        image.save(thumbnail_folder + thumbnail_prefix + image_filename)

    return image_filename
