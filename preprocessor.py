import re
import pandas as pd
from datetime import datetime


def preprocess(data):
    # Combined pattern for both iPhone and Android formats
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}[,\s]\d{1,2}:\d{2}(?::\d{2})?(?:\s[APap][Mm])?)\s*[-|\]]\s*'

    messages = []
    dates = []

    # Split the data using the pattern
    parts = re.split(pattern, data)

    # The first part is usually empty or metadata, so we skip it
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            dates.append(parts[i].strip())
            messages.append(parts[i + 1].strip())

    # If no dates found with the pattern, try alternative patterns
    if len(dates) == 0:
        # Try iPhone pattern
        iphone_pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[APap][Mm]?)\]'
        dates = re.findall(iphone_pattern, data)
        messages = re.split(iphone_pattern, data)[1:]
        messages = messages[1::2]  # Get every other element

    if len(dates) == 0:
        # Try Android pattern
        android_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[APap][Mm])?)'
        dates = re.findall(android_pattern, data)
        messages = re.split(android_pattern, data)[1:]
        messages = messages[1::2]  # Get every other element

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Function to parse dates with multiple formats
    def parse_date(date_str):
        date_str = str(date_str).strip()

        # Remove any brackets
        date_str = date_str.strip('[]')

        # Try different date formats
        formats = [
            '%d/%m/%y, %I:%M:%S %p',  # iPhone: 25/12/23, 10:30:00 AM
            '%d/%m/%Y, %I:%M:%S %p',  # iPhone: 25/12/2023, 10:30:00 AM
            '%d/%m/%y, %I:%M %p',  # Android: 25/12/23, 10:30 AM
            '%d/%m/%Y, %I:%M %p',  # Android: 25/12/2023, 10:30 AM
            '%m/%d/%y, %I:%M:%S %p',  # US format iPhone
            '%m/%d/%Y, %I:%M:%S %p',  # US format iPhone
            '%m/%d/%y, %I:%M %p',  # US format Android
            '%m/%d/%Y, %I:%M %p',  # US format Android
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If none work, return NaT
        return pd.NaT

    # Apply the parsing function
    df['date'] = df['message_date'].apply(parse_date)

    # Drop rows where date parsing failed
    df = df.dropna(subset=['date'])

    # Separate users and messages
    users = []
    clean_messages = []

    for message in df['user_message']:
        # Handle different separators
        if ':' in message:
            # Split on first colon
            parts = message.split(':', 1)
            if len(parts) == 2:
                users.append(parts[0].strip())
                clean_messages.append(parts[1].strip())
            else:
                users.append('group_notification')
                clean_messages.append(message.strip())
        else:
            # No user found (system messages, media omitted, etc.)
            if 'omitted' in message.lower() or 'media' in message.lower():
                users.append('system')
            else:
                users.append('group_notification')
            clean_messages.append(message.strip())

    df['user'] = users
    df['message'] = clean_messages
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    # Extract date components
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create period column
    period = []
    for hour in df['hour']:
        start = f"{hour:02d}"
        end = f"{(hour + 1) % 24:02d}"
        period.append(f"{start}-{end}")

    df['period'] = period

    # Clean up user names
    # Remove common WhatsApp artifacts
    df['user'] = df['user'].str.replace(r'[^\w\s]+', '', regex=True)
    df['user'] = df['user'].str.strip()

    # Replace empty user names
    df['user'] = df['user'].replace('', 'group_notification')

    return df
