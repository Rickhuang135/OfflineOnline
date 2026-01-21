class Words:
    SAVEGAME = "save" # Start saving every frame of game
    CLOSEDISPLAYS = "end" # Gracefully shutdown each display
    setVerbose = "verbose" # Instructs child to become verbose
    setSilent = "silent" # Instructs child to become silent
    id = "id" # Display identity for a child vgui process
    time_stamp = "time" # Time screenshot was taken (since start)
    shm_name = "shm_name" # Name of shared memory buffer
    shape = "shape" # Shape of shared memory buffer
    dtype = "dtype" # Datatype of shared memory buffer
    

class Actions: # keep these as the correct keys for pyautogui
    Jump = "up"
    Duck = "down"
    Nothing = "2"