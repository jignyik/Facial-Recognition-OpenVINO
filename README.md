# Yes
Face Recognition

Running on python 3.8
# Prerequisites

Pip install requirements

    pip install openvino opencv-python 

# Running the program
Run main.py

# Error with openvino (when running main.py)


    Traceback (most recent call last):
        File "C:/Users/chanj/Desktop/FER-FaceRecog-main/main.py", line 3, in <module>
            from school import Arguments, FrameProcessor, draw_detection
        File "C:\Users\chanj\Desktop\FER-FaceRecog-main\school.py", line 2, in <module>
            from openvino.inference_engine import IECore
        File "C:\ProgramData\Anaconda3\lib\site-packages\openvino\inference_engine\__init__.py", line 30, in <module>
            from .ie_api import *
    ImportError: DLL load failed while importing ie_api: The specified module could not be found.

1. Go to "C:\ProgramData\Anaconda3\lib\site-packages\openvino\inference_engine\__init__.py"
2. Copy line 28 

        os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]
3. Paste in line 26 before the "else" statement or copy the following code and paste in "__init__.py"

        import os
        import sys
        
        if sys.platform == "win32":
            # Installer, yum, pip installs openvino dlls to the different directories
            # and those paths need to be visible to the openvino modules
            #
            # If you're using a custom installation of openvino,
            # add the location of openvino dlls to your system PATH.
            #
            # looking for the libs in the pip installation path by default.
            openvino_libs = [os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")]
            # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
            openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
            if openvino_libs_installer:
                openvino_libs.extend(openvino_libs_installer.split(";"))
            for lib in openvino_libs:
                lib_path = os.path.join(os.path.dirname(__file__), lib)
                if os.path.isdir(lib_path):
                    # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
                    if (3, 8) <= sys.version_info:
                        os.add_dll_directory(os.path.abspath(lib_path))
                        os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]
                    else:
                        os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]
        
        from .ie_api import *
        
        __all__ = ["IENetwork", "TensorDesc", "IECore", "Blob", "PreProcessInfo", "get_version"]
        __version__ = get_version()  # type: ignore

