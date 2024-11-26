from data_preprocessor import preprocess_text  # Import the existing preprocessing function

# Sample test data
sample_texts = [
    None,  # None value
    "",  # Empty string
    "    ",  # String with only spaces
    12345,  # Numeric value
    ["list", "of", "words"],  # List input
    "This is a valid text!",  # Valid text
    "   Valid with spaces!   ",  # Valid text with leading and trailing spaces
    "http://example.com Check this out!",  # Text containing a URL
    "!!! Invalid #Text###"  # Text with special characters
]

# Function to run tests
def run_tests():
    for i, text in enumerate(sample_texts, 1):
        try:
            result = preprocess_text(text, remove_specials=True)  # Enable the option to remove special characters
            print(f"Test {i} - Original: {text} => Processed: {result}")
        except Exception as e:
            print(f"Test {i} - Original: {text} => Error: {e}")

if __name__ == "__main__":
    run_tests()
