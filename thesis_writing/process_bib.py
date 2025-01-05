import re

def parse_bib_file(file_path):
    """
    Parses a .bib file to extract URLs, DOIs, or titles when neither is found.

    Args:
        file_path (str): Path to the .bib file.

    Returns:
        None: Prints the output directly.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the .bib file into individual references
    entries = re.split(r'@\w+\{', content)[1:]  # Ignore the first split part as it's before the first @

    for entry in entries:
        # Extract URL, DOI, and title
        url_match = re.search(r'url\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)

        url = url_match.group(1) if url_match else None
        doi = doi_match.group(1) if doi_match else None
        title = title_match.group(1) if title_match else None

        if url:
            print(f"URL: {url}")
        elif doi:
            print(f"DOI: {doi}")
        elif title:
            print(f"Title: {title}")
        else:
            print("No URL, DOI, or Title found in this entry.")

if __name__ == "__main__":
    # Replace 'your_file.bib' with the path to your .bib file
    parse_bib_file("./sample.bib")
    
