from urlextract import URLExtract
import pandas as pd
import emoji
from collections import Counter
from wordcloud import WordCloud
extract = URLExtract()
def fetch_stats(selected_user,df):
    '''if selected_user == "Overall":
        # fetch the number of messages
        num_messages = df.shape[0]

        # fetch the total number of words
        words = []
        for message in df['message']:
            words.extend(message.split())
        return num_messages, len(words)
    else:
        new_df = df[df['user'] == selected_user]

        # fetch the number of messages
        num_messages = new_df.shape[0]

        # fetch the total number of words
        words = []
        for message in new_df['message']:
            words.extend(message.split())
        return num_messages, len(words)'''

    # same code as above but in structured way
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

        # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    #num_media_messages = df[df['message'] == '‎image omitted\r\n'].shape[0]
    num_media_messages = df[
        (df['message'].str.contains('omitted', case=False, na=False)) |
        (df['message'].str.contains('media', case=False, na=False)) |
        (df['message'] == '‎image omitted\r\n') |
        (df['message'] == '<Media omitted>')
        ].shape[0]
    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x,df

# to form word cloud
def create_wordcloud(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

#most commonly used words
def most_common_words(selected_user,df):

    #f = open('stop_hinglish.txt','r')
    #stop_words = f.read()
    
    
    from pathlib import Path
    BASE_DIR = Path(__file__).parent
    stop_words_path = BASE_DIR / "stop_hinglish.txt"

    with open(stop_words_path, "r", encoding="utf-8") as f:
        stop_words = f.read()

    
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

#emoji analysis
def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

#def activity_heatmap(selected_user,df):

    #if selected_user != 'Overall':
        #df = df[df['user'] == selected_user]

    #user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)


    #return user_heatmap
def activity_heatmap(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    #  NEW: handle empty dataframe
    if df.empty:
        return None

    user_heatmap = (
        df.pivot_table(
            index='day_name',
            columns='period',
            values='message',
            aggfunc='count'
        )
        .fillna(0)
    )

    # NEW: handle all-zero heatmap
    if user_heatmap.sum().sum() == 0:
        return None

    return user_heatmap



