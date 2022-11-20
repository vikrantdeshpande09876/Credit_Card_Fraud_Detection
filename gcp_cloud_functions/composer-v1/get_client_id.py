'''
    ONE TIME EXECUTION: RUN THIS SCRIPT IN A GOOGLE CLOUD CONSOLE SHELL
'''


# This script is intended to be used with Composer 1 environments
# In Composer 2, the Airflow Webserver is not in the tenant project
# so there is no tenant client ID
# See https://cloud.google.com/composer/docs/composer-2/environment-architecture
# for more details
import google.auth
import google.auth.transport.requests
import requests
import six.moves.urllib.parse

# Authenticate with Google Cloud.
# See: https://cloud.google.com/docs/authentication/getting-started
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
authed_session = google.auth.transport.requests.AuthorizedSession(credentials)


# CONFIGURE THESE PARAMETERES
project_id = 'duke-energy-big-data-project'
location = 'us-east1'
composer_environment = 'i535-airflow-orch'

environment_url = (
    "https://composer.googleapis.com/v1beta1/projects/{}/locations/{}"
    "/environments/{}"
).format(project_id, location, composer_environment)

composer_response = authed_session.request("GET", environment_url)
environment_data = composer_response.json()
composer_version = environment_data["config"]["softwareConfig"]["imageVersion"]
if "composer-1" not in composer_version:
    version_error = ("This script is intended to be used with Composer 1 environments. "
                     "In Composer 2, the Airflow Webserver is not in the tenant project, "
                     "so there is no tenant client ID. "
                     "See https://cloud.google.com/composer/docs/composer-2/environment-architecture for more details.")
    raise (RuntimeError(version_error))
    

# CONFIGURE THESE PARAMETERES
if 'airflowUri' in environment_data['config']:
    airflow_uri = environment_data["config"]["airflowUri"]
else:
    airflow_uri = 'https://lc671f9a53c03e59cp-tp.appspot.com'


# The Composer environment response does not include the IAP client ID.
# Make a second, unauthenticated HTTP request to the web server to get the
# redirect URI.
redirect_response = requests.get(airflow_uri, allow_redirects=False)
redirect_location = redirect_response.headers["location"]

# Extract the client_id query parameter from the redirect.
parsed = six.moves.urllib.parse.urlparse(redirect_location)
query_string = six.moves.urllib.parse.parse_qs(parsed.query)
print(query_string["client_id"][0])