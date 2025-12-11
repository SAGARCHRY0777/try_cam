import cv2
import numpy as np
from pypylon import pylon
from ..camera_class import Camera

class BaslerGigECamera(Camera):

    def __init__(self, config: dict):
        # Create a camera object using the provided IP address
        print("2")
        super().__init__(config)

        self.cam_ip = config.get('cam_ip', '')   # get IP address from config dict
        self.device = config.get('type', 'BaslerGigE')

        #! Connecting basler camera with IpAdress
        # ip_address = "169.254.1.1"  # Replace with the actual IP address of the camera you want to connect to
        # ip_address = ""  # Replace with the actual IP address of the camera you want to connect to
        # ip_address = "172.17.0.106"  # Replace with the actual IP address of the camera you want to connect to

        # Create a device filter
        di = pylon.DeviceInfo()
        di.SetDeviceClass(self.device)  # Use the appropriate device class for GigE cameras
        di.SetIpAddress(self.cam_ip)

        # Create an InstantCamera object
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))
        self.thread_run = True
        
        self.camera.Open()
        
        self.thread_run = True
        self.last_ret = False
        print("3")
    
    def connect(self):
        connection_established = self.camera.IsOpen()

        if connection_established:
            print("Hi mubarak Basler connected")
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImages)

            # self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("Camera is connected",connection_established)
            self._apply_config(self.config)

            return True
        else:
            print('Unable to open camera',connection_established)
            return False
        
    def disconnect(self):
        try:
            self.camera.Close()
            return True
        except Exception as e:
            print("Error:", e)
            return False
        
    def fetch_frame(self):
        if not self.camera.IsOpen():
            print('<ERROR> Camera Not Connected:', self.cam_ip)
            return None

        try:
            # self.camera.Open()
            #self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            
            # Execute software trigger
            # self.camera.ExecuteSoftwareTrigger()
            # self.camera.TriggerSelector = "FrameStart"
            # self.camera.GenerateSoftwareTrigger ="Execute"
            # print("Before execution")
            # self.camera.GenerateSoftwareTrigger.SetValue("Execute")
            # print("After execution")
            # self.camera.StartGrabbing(pylon.GrabStrategy_LatestImages)
            # if self.camera.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
            #         self.camera.ExecuteSoftwareTrigger()

            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    img_data = grabResult.Array
                    img_np = img_data.astype('uint8')
                    self.latest_frame = img_np
                    print("frammmmmmmmmm",self.latest_frame)
                    return self.latest_frame
                else:
                    print('Camera Buffer Empty:', self.cam_ip)
                    return None

        except Exception as e:
            print("Error while grabbing single frame ::", e)
            return None


    def _apply_config(self, updates: dict):
        if not self.camera or not self.camera.IsOpen():
            print("<ERROR> Camera not open")
            return

        try:
            # --- Exposure ---
            if "exposure" in updates:
                if str(updates["exposure"]).lower() == "auto":
                    self.change_exposure("auto")
                else:
                    self.change_exposure(int(updates["exposure"]))  # Will call ExposureAuto=Off internally

            # --- Gain ---
            if "gain" in updates:
                current_gain = self.get_gain()
                print(f"Using current camera gain: {current_gain}")
                # Do nothing else — just read and possibly log it

            # --- White Balance ---
            if "white_balance_auto" in updates:
                if updates["white_balance_auto"] == "once":
                    self.set_colorbalance_once()
                elif updates["white_balance_auto"] is True or updates["white_balance_auto"] == "continuous":
                    self.camera.BalanceWhiteAuto.SetValue("Continuous")
                else:
                    self.disable_colorbalance()

            # --- Resolution ---
            if "width" in updates and "height" in updates:
                self.camera.Width.SetValue(int(updates["width"]))
                self.camera.Height.SetValue(int(updates["height"]))

            print("Camera configuration applied successfully.")

        except Exception as e:
            print(f"<ERROR> Failed to apply config: {e}")

    def SetExposureTime(self,value):
            connection_established=self.camera.IsOpen()

            if connection_established:
                self.camera.ExposureTimeAbs.SetValue(value)
                return 1
            else:
                print('Unable to open camera',connection_established)
                return 0
            
    def change_exposure(self, exposure):
        try:
            if exposure == 'auto':
                # Set the Exposure Auto auto function to its minimum lower limit
                # and its maximum upper limit
                min_lower_limit = self.camera.AutoExposureTimeLowerLimitRaw.GetMin()
                max_upper_limit = self.camera.AutoExposureTimeUpperLimitRaw.GetMax()
                self.camera.AutoExposureTimeLowerLimitRaw.SetValue(min_lower_limit)
                self.camera.AutoExposureTimeUpperLimitRaw.SetValue(max_upper_limit)

                # Set the target brightness value to 128
                self.camera.AutoTargetValue.SetValue(128)

                # Select auto function ROI 1
                self.camera.AutoFunctionAOISelector.SetValue("AOI1")

                # Enable the 'Intensity' auto function (Gain Auto + Exposure Auto)
                # for the auto function ROI selected
                self.camera.AutoFunctionAOIUsageIntensity.SetValue(True)

                # Enable Exposure Auto by setting the operating mode to Continuous
                self.camera.ExposureAuto.SetValue("Continuous")
                # Set the ExposureMode parameter to Timed
                exposure_mode = self.camera.ExposureMode  # Replace with the correct property name
                exposure_mode.SetValue("Timed")
                print('Camera Exposure Changed to : ',exposure)
            else:
                self.camera.ExposureAuto.SetValue("Off")
                exposure_time_abs = self.camera.ExposureTimeAbs  # Replace with the correct property name
                exposure_time_abs.SetValue(int(exposure))
                print('Camera Exposure Changed to : ',exposure)
            return True
        except Exception as ex :
            print('<ERROR> Unable to Change Camera Exposure : ', ex)
            return False
        
    def disable_colorbalance(self):
        try:
            self.camera.BalanceWhiteAuto.SetValue("Off")
            # self.camera.BalanceWhiteAutoContinuous.SetValue("Off")
            print('Camera Color Balance Changed to Off')
            return True
        except Exception as ex :
            print('<ERROR> Unable to Change Camera Color Balance : ', ex)
            return False

    def set_colorbalance_once(self):
        try:
            # Set the BalanceWhiteAuto parameter to On
            # self.camera.BalanceWhiteAuto.SetValue("On")

            # Enable Balance White Auto by setting the operating mode to Continuous
            # self.camera.BalanceWhiteAutoContinuous.SetValue("On")
            self.camera.BalanceWhiteAuto.SetValue("Once")
            print('Camera Color Balance Changed to Once')
            return True
        except Exception as ex :
            print('<ERROR> Unable to Change Camera Color Balance : ', ex)
            return False

    def get_gain(self):
        if self.camera.IsOpen():
            current_gain_value = self.camera.GainRaw.GetValue()
            return current_gain_value
        else:
            return False


