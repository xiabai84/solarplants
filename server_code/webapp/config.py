class Config(object):
    SITE_NAME='Solar'
    SITE_SLUG_NAME='Solar'
    SITE_LOCATION='Outer space'
    ADMINS=['']


class DevelopmentConfig(Config):
    DOMAIN= 'localhost=5000'
    ASSET_DOMAIN= 'localhost=5000'

class ProductionConfig(Config):
    DOMAIN= 'https://solarplant.herokuapp.com/'
    ASSET_DOMAIN= 'https://solarplant.herokuapp.com/'
