import axios from "axios";
import Swal from "sweetalert2";
import { TOKEN_STORAGE_KEY, OPENID_SIGNIN_URL } from "assets/config/const";

const instance = axios.create();

const loginURL = window.localStorage.getItem(OPENID_SIGNIN_URL) || "";

// Request interceptor for API calls
instance.interceptors.request.use(
  async (config) => {
    const apiToken = window.localStorage.getItem(TOKEN_STORAGE_KEY);
    config.headers = {
      Authorization: `Bearer ${apiToken}`,
      "Content-Type": "application/json",
    };
    return config;
  },
  (error) => {
    Promise.reject(error);
  }
);

// Response interceptor for API calls
instance.interceptors.response.use(
  (response) => {
    console.info("response:", response);
    if (response.status === 401 || response.status === 403) {
      // Redirect to Authing Login
      window.location.href = loginURL;
      return Promise.reject("401 User Unauthorized");
    } else {
      if (response) {
        return Promise.resolve(response);
      } else {
        return Promise.reject("response error");
      }
    }
  },
  (error) => {
    console.info("ERR:", error.response);
    // Swal.fire(error.message);
    Swal.fire(
      `${error.message}`,
      `${error.response?.config?.url} \n ${
        error.response?.config?.params
          ? JSON.stringify(error?.response?.config?.params)
          : ""
      }${error?.response?.data?.Message}`,
      undefined
    );
    console.log("-- error --");
    console.error(error);
    console.log("-- error --");
    return Promise.reject({
      success: false,
      msg: error,
    });
  }
);

export default instance;
