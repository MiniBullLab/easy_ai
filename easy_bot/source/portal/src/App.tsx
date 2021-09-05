import React, { useEffect, Suspense, useState } from "react";
import axios from "axios";
import jwt_decode from "jwt-decode";
import { useTranslation } from "react-i18next";
import { withTranslation } from "react-i18next";
import { HashRouter, Route } from "react-router-dom";
import { useDispatch } from "redux-react-hook";

import Amplify from "aws-amplify";
import { AmplifyAuthenticator, AmplifySignIn } from "@aws-amplify/ui-react";
import { AuthState, onAuthUIStateChange } from "@aws-amplify/ui-components";

import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import Button from "@material-ui/core/Button";
import Typography from "@material-ui/core/Typography";

import DataLoading from "common/Loading";
import TopBar from "common/TopBar";

import Home from "pages/home/Home";
import StepOne from "pages/creation/StepOne";
import StepTwoImage from "pages/creation/image/StepTwoImage";
import StepTwoObject from "pages/creation/object/StepTwoObject";
import StepTwoNER from "pages/creation/ner/StepTwoNER";
import TaskList from "pages/list/TaskList";
import {
  OPEN_ID_TYPE,
  API_URL_NAME,
  TOKEN_STORAGE_KEY,
  ID_TOKEN_STORAGE_KEY,
  OPENID_SIGNOUT_URL,
  OPENID_SIGNIN_URL,
  AUTH_TYPE_NAME,
} from "assets/config/const";
import { ACTION_TYPES } from "store/types";

const HomePage = withTranslation()(Home);
const StepOnePage = withTranslation()(StepOne);
const StepTwoImagePage = withTranslation()(StepTwoImage);
const StepTwoObjectPage = withTranslation()(StepTwoObject);
const StepTwoNERPage = withTranslation()(StepTwoNER);
const TaskListPage = withTranslation()(TaskList);

// loading component for suspense fallback
const Loader = () => (
  <div className="App">
    <div className="app-loading">
      <DataLoading />
      Machine Learning Bot is loading...
    </div>
  </div>
);

const getUrlToken = (name: string, str: string) => {
  const reg = new RegExp(`(^|&)${name}=([^&]*)(&|$)`);
  const r = str.substr(1).match(reg);
  if (r != null) return decodeURIComponent(r[2]);
  return "";
};

interface AwsConfigType {
  authType: string;
  apiUrl: string;
  aws_oidc_token_validation_url: string;
  aws_oidc_logout_url: string;
  aws_oidc_login_url: string;
  aws_oidc_provider: string;
}

const App: React.FC = () => {
  const dispatch = useDispatch();
  const { t } = useTranslation();

  const [loginUrl, setLoginUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [tokenIsValid, setTokenIsValid] = useState(true);
  const [authType, setAuthType] = useState<string>("");
  const [authState, setAuthState] = useState<AuthState>();
  const [user, setUser] = useState<any | undefined>();
  const [oidcIssuer, setOidcIssuer] = useState("");

  useEffect(() => {
    const timeStamp = new Date().getTime();
    axios.get("/aws-exports.json?timeStamp=" + timeStamp).then((res) => {
      const ConfigObj: AwsConfigType = res.data;
      const API_URL = ConfigObj.apiUrl;
      dispatch({ type: ACTION_TYPES.SET_API_URL, apiUrl: API_URL });
      const AuthType = ConfigObj.authType;
      localStorage.setItem(AUTH_TYPE_NAME, AuthType);
      setAuthType(AuthType);
      Amplify.configure(ConfigObj);
      if (ConfigObj.authType === OPEN_ID_TYPE) {
        const AWS_OIDC_TOKEN_VALIDATION_URL =
          ConfigObj.aws_oidc_token_validation_url;
        const AWS_OIDC_LOGOUT_URL = ConfigObj.aws_oidc_logout_url;
        localStorage.setItem(OPENID_SIGNOUT_URL, AWS_OIDC_LOGOUT_URL);
        const AWS_OIDC_LOGIN_URL = ConfigObj.aws_oidc_login_url;
        localStorage.setItem(OPENID_SIGNIN_URL, AWS_OIDC_LOGIN_URL);
        setLoginUrl(AWS_OIDC_LOGIN_URL);
        setOidcIssuer(ConfigObj.aws_oidc_provider);
        // Get Access Token if exsit
        const token = getUrlToken("access_token", window.location.hash);
        // Get Id Token if exsit
        const id_token = getUrlToken("id_token", window.location.hash);
        // If token exsit, set to localStorage, and then redirect to /
        if (token) {
          localStorage.setItem(TOKEN_STORAGE_KEY, token);
          localStorage.setItem(ID_TOKEN_STORAGE_KEY, id_token);
          window.location.href = "/";
        } else {
          // get token from localstorage
          const curToken = localStorage.getItem(TOKEN_STORAGE_KEY);
          if (curToken) {
            // if got token to validate it
            if (AWS_OIDC_TOKEN_VALIDATION_URL) {
              axios
                .get(
                  AWS_OIDC_TOKEN_VALIDATION_URL + "?access_token=" + curToken
                )
                .then((res) => {
                  setLoading(false);
                  if (res.data.iss) {
                  } else {
                    window.location.href = AWS_OIDC_LOGIN_URL;
                  }
                })
                .catch((err) => {
                  console.error(err);
                  // setLoading(false);
                  window.location.href = AWS_OIDC_LOGIN_URL;
                });
            }
          } else {
            // setLoading(false);
            window.location.href = AWS_OIDC_LOGIN_URL;
          }
        }
      } else {
        setLoading(false);
      }
      window.localStorage.setItem(API_URL_NAME, API_URL);
      // setLoading(false);
    });
  }, [dispatch]);

  useEffect(() => {
    return onAuthUIStateChange((nextAuthState, authData: any) => {
      setAuthState(nextAuthState);
      setUser(authData);
      console.info("authDataauthData:", authData);
      localStorage.setItem(
        TOKEN_STORAGE_KEY,
        authData?.signInUserSession?.accessToken?.jwtToken || ""
      );
      if (authData && authData.hasOwnProperty("attributes")) {
        localStorage.setItem("authDataEmail", authData.attributes.email);
      }
    });
  }, []);

  // Check Token Expire when OPENID
  useEffect(() => {
    const interval = setInterval(() => {
      const curToken = localStorage.getItem(TOKEN_STORAGE_KEY);
      if (curToken) {
        const myDecodedToken: any = jwt_decode(curToken);
        if (myDecodedToken.exp * 1000 < new Date().getTime()) {
          setTokenIsValid(false);
        } else {
          if (authType === OPEN_ID_TYPE) {
            // Check Issuer Correct
            if (oidcIssuer === myDecodedToken.iss) {
              setTokenIsValid(true);
            } else {
              setTokenIsValid(false);
            }
          } else {
            setTokenIsValid(true);
          }
        }
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [authType, oidcIssuer]);

  if (loading) {
    return <Loader />;
  }
  return authType === OPEN_ID_TYPE ||
    (authState === AuthState.SignedIn && user) ? (
    <div className="bp3-dark">
      {!tokenIsValid && (
        <div className="over-mask">
          <div className="card">
            <Card>
              <CardContent>
                <Typography
                  style={{
                    borderBottom: "1px solid #ddd",
                    paddingBottom: "10px",
                  }}
                  color="textSecondary"
                  gutterBottom
                >
                  {t("reLogin")}
                </Typography>
                <Typography variant="h5" component="h2"></Typography>
                <Typography variant="body2" component="p">
                  {t("reLoignTips")}
                </Typography>
              </CardContent>
              <div className="text-right relogin-botton">
                <Button
                  onClick={() => {
                    window.location.href = loginUrl;
                  }}
                  variant="contained"
                  color="primary"
                >
                  {t("btn.reLogin")}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      )}
      <TopBar />
      <React.Fragment>
        <HashRouter>
          <Route path="/" exact component={HomePage}></Route>
          <Route path="/home" exact component={HomePage}></Route>
          <Route path="/create/step1" exact component={StepOnePage}></Route>
          <Route
            path="/create/step2/image_classification/:id"
            exact
            component={StepTwoImagePage}
          ></Route>
          <Route
            path="/create/step2/object_detection/:id"
            exact
            component={StepTwoObjectPage}
          ></Route>
          <Route
            path="/create/step2/named_entity_recognition/:id"
            exact
            component={StepTwoNERPage}
          ></Route>
          <Route path="/task-list" exact component={TaskListPage}></Route>
        </HashRouter>
      </React.Fragment>
    </div>
  ) : (
    <div className="login-wrap">
      <AmplifyAuthenticator>
        <AmplifySignIn
          headerText="Sign in to Machine Learning Bot"
          slot="sign-in"
          usernameAlias="username"
          formFields={[
            {
              type: "username",
              label: "Email *",
              placeholder: "Enter your email",
              required: true,
              inputProps: { autoComplete: "off" },
            },
            {
              type: "password",
              label: "Password *",
              placeholder: "Enter your password",
              required: true,
              inputProps: { autoComplete: "off" },
            },
          ]}
        >
          {/* <div slot="secondary-footer-content"></div> */}
        </AmplifySignIn>
      </AmplifyAuthenticator>
    </div>
  );
};

const WithProvider = () => <App />;

// here app catches the suspense from page in case translations are not yet loaded
export default function RouterApp(): JSX.Element {
  return (
    <Suspense fallback={<Loader />}>
      <WithProvider />
    </Suspense>
  );
}
