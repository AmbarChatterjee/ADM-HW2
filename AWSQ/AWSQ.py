import pandas as pd
import json
from collections import Counter
import time

# Open the JSON file in read mode
json_objects = []
with open('list.json', 'r') as f:
    for line in f:
        json_objects.append(json.loads(line))

# Extract 'tags' column from each JSON object
tags_data = [obj['tags'] for obj in json_objects if 'tags' in obj]

# Load 'tags' data into a pandas DataFrame
df = pd.DataFrame({'tags': tags_data})

# Start the timer
start_time = time.time()

# Create a Counter object to count the occurrences
tag_counter = Counter()

# Handle missing values by filling with an empty list
df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

# Remove unwanted characters, convert to lowercase, and update the counter
for tags in df['tags']:
    cleaned_tags = [tag.lower().replace(' ', '') for tag in tags]
    tag_counter.update(cleaned_tags)

# Get the five most common tags
top_five_tags = tag_counter.most_common(5)

# End the timer
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

# Create DataFrame for the top five tags
df_tags = pd.DataFrame(top_five_tags, columns=['tag', '#usage'])

# Print the DataFrames
print(df_tags)

# Print the time taken
print(f"\nTime taken: {time_taken} seconds")