import Cocoa
import sys

def set_icon(image_path, file_path):
    image = Cocoa.NSImage.alloc().initWithContentsOfFile_(image_path)
    if image:
        Cocoa.NSWorkspace.sharedWorkspace().setIcon_forFile_options_(image, file_path, 0)
        print("Icon set successfully")
    else:
        print("Failed to load image")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_icon.py <image_path> <file_path>")
    else:
        set_icon(sys.argv[1], sys.argv[2])
