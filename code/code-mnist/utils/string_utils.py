import re

def replace_last_number_with_new_number(input_string, new_number):
    # Use regular expression to find the last number in the string
    match = re.search(r'\d+$', input_string)

    if match:
        # Replace the last number with the new number
        modified_string = input_string[:match.start()] + str(new_number)
        return modified_string
    else:
        # Handle cases where there is no number at the end of the string
        return input_string