from pathlib import Path
c_file = Path(__file__).resolve()
Images = c_file.parent.parent.parent / "Images"

Screenshots = Images / "Screenshots"
locateOnScreenTargets = Images / "Targets"
crashRecords = Images / "CrashRecords"
VirtualUsers = c_file.parent / "VirtualUsers"