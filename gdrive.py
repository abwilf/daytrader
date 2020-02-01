import pickle
from os import path, remove
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_service(stocks_path):
    creds = None
    if path.exists(f'{stocks_path}token.pickle'):
        with open(f'{stocks_path}token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                f'{stocks_path}credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(f'{stocks_path}token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

def get_file(file_id, stocks_path):
    service = get_service(stocks_path)
    return service.files().get_media(fileId=file_id).execute().decode("utf-8")

def replace_files(files, stocks_path):
    ''' Replace local files with copies in stocks_path '''
    for filename, file_id in files:
        if path.exists(filename):
            remove(filename)
        with open(filename, 'w') as f:
            f.write(get_file(file_id, stocks_path))