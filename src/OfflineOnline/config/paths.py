from pathlib import Path
this_file = Path(__file__).resolve()
Root = this_file.parent.parent

config = this_file.parent

Images = Root / "Images"
Screenshots = Images / "Screenshots"
locateOnScreenTargets = Images / "Targets"
crashRecords = Images / "CrashRecords"

environment = Root / "environment"
VirtualUsers = environment / "VirtualUsers"

