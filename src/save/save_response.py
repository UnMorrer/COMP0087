import os

def response(string, file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)

    try:
        if not os.path.exists(folder_path):
            raise Exception(f"Folder '{folder_path}' does not exist")

        with open(file_path, 'w') as file:
            file.write(string)

    except Exception as e:
        print(f"An error occurred while trying to save the string to a file: {e}")
