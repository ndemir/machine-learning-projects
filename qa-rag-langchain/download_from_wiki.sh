#!/bin/bash

# Array of all US states
states=("Alabama" "Alaska" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida"
"Georgia" "Hawaii" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland"
"Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire"
"New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Oregon" "Pennsylvania"
"Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington"
"West Virginia" "Wisconsin" "Wyoming")

mkdir -p wiki
# Loop through each state and download its Wikipedia page
for state in "${states[@]}"; do
    # Replace spaces with underscores for the URL and the filename
    formatted_state=$(echo $state | sed 's/ /_/g')

    # Form the Wikipedia URL
    url="https://en.wikipedia.org/wiki/$formatted_state"

    # Download the page content using lynx and save it to a file with underscores in the name
    lynx -dump -nolist "$url" > "wiki/${formatted_state}.txt"
done

echo "Download complete!"
