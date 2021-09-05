import React, { useEffect, useCallback, useState } from "react";
import { useDispatch, useMappedState } from "redux-react-hook";
import { useHistory, Link } from "react-router-dom";
import classNames from "classnames";
import Loader from "react-loader-spinner";
import { useTranslation } from "react-i18next";
import Axios from "assets/config/http";
import Moment from "react-moment";

import Loading from "common/Loading";
import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Typography from "@material-ui/core/Typography";
import MLink from "@material-ui/core/Link";
import RefreshIcon from "@material-ui/icons/Refresh";
import { SnackbarOrigin } from "@material-ui/core/Snackbar";
import Dialog from "@material-ui/core/Dialog";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import DialogTitle from "@material-ui/core/DialogTitle";
// import Pagination from "@material-ui/lab/Pagination";
import { withStyles } from "@material-ui/core/styles";
import Menu, { MenuProps } from "@material-ui/core/Menu";
import MenuItem from "@material-ui/core/MenuItem";
import ListItemText from "@material-ui/core/ListItemText";

import { IState } from "store/Store";

import LeftMenu from "common/LeftMenu";
import Bottom from "common/Bottom";
import InfoBar from "common/InfoBar";

import NormalButton from "common/comp/NormalButton";
import PrimaryButton from "common/comp/PrimaryButton";
import StopButtonLoading from "common/comp/PrimaryButtonLoading";

import STATUS_PENDING from "@material-ui/icons/Schedule";
import STATUS_ERROR from "@material-ui/icons/HighlightOff";
import STATUS_DONE from "@material-ui/icons/CheckCircleOutline";
import STATUS_NOTSTART from "@material-ui/icons/NotInterested";
import STATUS_HOSTING from "@material-ui/icons/BlurCircular";

import "./TaskList.scss";

import STATUS_OK from "@material-ui/icons/CheckCircleOutline";

import PAGE_PREV from "@material-ui/icons/NavigateBefore";
import PAGE_PREV_DISABLED from "@material-ui/icons/NavigateBefore";
import PAGE_NEXT from "@material-ui/icons/NavigateNext";
import PAGE_NEXT_DISABLED from "@material-ui/icons/NavigateNext";

import { TASK_STATUS_MAP, TYPE_LIST } from "assets/types/index";

import {
  CUR_SUPPORT_LANGS,
  API_URL_NAME,
  URL_ML_IMAGE_TASKS,
  converListToMap,
} from "assets/config/const";
import { ACTION_TYPES } from "store/types";

export interface State extends SnackbarOrigin {
  open: boolean;
}

const mapState = (state: IState) => ({
  createTaskFlag: state.createTaskFlag,
  apiUrl: state.apiUrl,
});

const StyledMenu = withStyles({
  paper: {
    border: "1px solid #d3d4d5",
  },
})((props: MenuProps) => (
  <Menu
    style={{ borderRadius: 0 }}
    elevation={0}
    getContentAnchorEl={null}
    anchorOrigin={{
      vertical: "bottom",
      horizontal: "left",
    }}
    transformOrigin={{
      vertical: "top",
      horizontal: "left",
    }}
    {...props}
  />
));

const TYPE_LIST_MAP = converListToMap(TYPE_LIST);

console.info("TYPE_LIST_MAP:", TYPE_LIST_MAP);

const StyledMenuItem = withStyles((theme) => ({
  root: {
    width: 130,
    "& .MuiTypography-body1": {
      fontSize: 14,
    },
  },
}))(MenuItem);

const STATUS_ICON_MAP: any = {
  NotStarted: <STATUS_NOTSTART fontSize="small" />,
  Training: <STATUS_PENDING fontSize="small" />,
  Failed: <STATUS_ERROR fontSize="small" />,
  Hosting: <STATUS_HOSTING fontSize="small" />,
  Completed: <STATUS_DONE fontSize="small" />,
};

const List: React.FC = () => {
  const { t, i18n } = useTranslation();
  const dispatch = useDispatch();

  const [nameStr, setNameStr] = useState("en_name");

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setNameStr(i18n.language + "_name");
    }
  }, [i18n.language]);

  const { apiUrl, createTaskFlag } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);

  const history = useHistory();
  const [isLoading, setIsLoading] = useState(true);
  const [isStopLoading, setIsStopLoading] = useState(false);
  const [curPage, setCurPage] = useState(1);
  const [isLastpage, setIsLast] = useState(false);
  const [taskListData, setTaskListData] = useState<any>([]);
  const [curSelectTask, setCurSelectTask] = useState<any>(null);
  const [open, setOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [curActionType, setCurActionType] = useState("");

  // Hide Create Flag in 3 seconds
  useEffect(() => {
    dispatch({
      type: ACTION_TYPES.CLOSE_SIDE_BAR,
    });
  }, [dispatch]);

  const goToStepOne = () => {
    dispatch({ type: ACTION_TYPES.CLOSE_SIDE_BAR });
    const toPath = "/create/step1";
    history.push({
      pathname: toPath,
    });
  };

  const clickTaskInfo = (taskInfo: any, event: any) => {
    setCurSelectTask(taskInfo);
  };

  const changeRadioSelect = (event: any) => {
    console.info("event:", event);
  };

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
  };

  const stopCurTask = (event: any) => {
    setAnchorEl(null);
    setCurActionType("STOP");
    setOpen(true);
  };

  const deleteCurTask = (event: any) => {
    setAnchorEl(null);
    setCurActionType("DELETE");
    setOpen(true);
  };

  const confirmDeleteTask = () => {
    setIsStopLoading(true);
    Axios.delete(API_URL + `tasks/${curSelectTask.taskName}`)
      .then((res) => {
        setIsStopLoading(false);
        getTaskList();
      })
      .catch((err) => {
        setIsStopLoading(false);
        console.error(err);
      });
  };

  const confirmStopTask = () => {
    setIsStopLoading(true);
    Axios.post(API_URL + `tasks/${curSelectTask.taskName}/stop`)
      .then((res) => {
        setOpen(false);
        setIsStopLoading(false);
        getTaskList();
      })
      .catch((err) => {
        setIsStopLoading(false);
        console.error(err);
      });
  };

  const getTaskList = useCallback(() => {
    // setIsLoading(true);
    Axios.get(API_URL + URL_ML_IMAGE_TASKS)
      .then((res) => {
        setCurSelectTask(null);
        setOpen(false);
        setIsLoading(false);
        const orderedList = res.data?.tasks || [];
        orderedList.sort((a: any, b: any) =>
          a.createTime < b.createTime ? 1 : -1
        );
        setTaskListData(orderedList);
        // If No next token, set is last page
        if (!res.data.nextToken) {
          setIsLast(true);
        }
      })
      .catch((err) => {
        console.error(err);
      });
  }, [API_URL]);

  const refreshData = () => {
    setCurPage(1);
    getTaskList();
  };

  useEffect(() => {
    getTaskList();
  }, [getTaskList]);

  return (
    <div className="drh-page">
      <Dialog
        open={open}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {t("taskList.stopTask")}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            {curActionType === "STOP" && t("taskList.tips.confimStop")}{" "}
            {curActionType === "DELETE" && t("taskList.tips.confirmDelete")}{" "}
            <b>{curSelectTask && curSelectTask.taskName}</b>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          {curActionType === "STOP" && (
            <div className="padding-15">
              <NormalButton
                onClick={() => {
                  setOpen(false);
                }}
                color="primary"
              >
                {t("btn.cancel")}
              </NormalButton>
              {isStopLoading ? (
                <StopButtonLoading disabled={true}>
                  <Loader type="ThreeDots" color="#ffffff" height={10} />
                </StopButtonLoading>
              ) : (
                <PrimaryButton
                  onClick={confirmStopTask}
                  color="primary"
                  autoFocus
                >
                  {t("btn.confirm")}
                </PrimaryButton>
              )}
            </div>
          )}
          {curActionType === "DELETE" && (
            <div className="padding-15">
              <NormalButton
                onClick={() => {
                  setOpen(false);
                }}
                color="primary"
              >
                {t("btn.cancel")}
              </NormalButton>
              {isStopLoading ? (
                <StopButtonLoading disabled={true}>
                  <Loader type="ThreeDots" color="#ffffff" height={10} />
                </StopButtonLoading>
              ) : (
                <PrimaryButton
                  onClick={confirmDeleteTask}
                  color="primary"
                  autoFocus
                >
                  {t("btn.confirm")}
                </PrimaryButton>
              )}
            </div>
          )}
        </DialogActions>
      </Dialog>

      <LeftMenu />
      <div className="right">
        <InfoBar />
        {createTaskFlag && (
          <div className="task-status">
            <div className="content">
              <STATUS_OK className="icon" />
              {t("taskList.tips.successMsg")}
            </div>
          </div>
        )}

        <div className="padding-right-40">
          <div className="page-breadcrumb">
            <Breadcrumbs
              separator={<NavigateNextIcon fontSize="small" />}
              aria-label="breadcrumb"
            >
              <MLink color="inherit" href="/#/">
                {t("breadCrumb.home")}
              </MLink>
              <Typography color="textPrimary">
                {t("breadCrumb.tasks")}
              </Typography>
            </Breadcrumbs>
          </div>
          <div className="table-data">
            <div className="box-shadow">
              <div className="title">
                <div className="options">
                  <div className="task-count">
                    {t("taskList.title")}
                    {/* <span className="info">(10)</span> */}
                  </div>
                  <div className="buttons">
                    <NormalButton onClick={refreshData}>
                      <RefreshIcon width="10" />
                    </NormalButton>
                    <div style={{ display: "inline-block" }}>
                      <NormalButton
                        disabled={curSelectTask === null}
                        aria-controls="customized-menu"
                        onClick={handleClick}
                      >
                        {t("btn.taskAction")}
                        <span style={{ marginLeft: 3 }}>▼</span>
                      </NormalButton>
                      <StyledMenu
                        id="customized-menu"
                        anchorEl={anchorEl}
                        keepMounted
                        open={Boolean(anchorEl)}
                        onClose={handleCloseMenu}
                      >
                        {!(curSelectTask === null) && (
                          <StyledMenuItem>
                            <ListItemText
                              onClick={stopCurTask}
                              primary={t("btn.stopTask")}
                            />
                          </StyledMenuItem>
                        )}
                        <StyledMenuItem>
                          <ListItemText
                            onClick={deleteCurTask}
                            primary={t("btn.deleteTask")}
                          />
                        </StyledMenuItem>
                      </StyledMenu>
                    </div>
                    <PrimaryButton onClick={goToStepOne}>
                      {t("btn.createTask")}
                    </PrimaryButton>
                  </div>
                </div>
                <div className="search">
                  <div className="search-input">
                    {/* <input type="text" placeholder="Find resources" /> */}
                  </div>
                  <div className="pagination">
                    <div>
                      {curPage > 1 && !isLoading ? (
                        <span className="item prev">
                          <PAGE_PREV />
                        </span>
                      ) : (
                        <span className="item prev disabled">
                          <PAGE_PREV_DISABLED color="disabled" />
                        </span>
                      )}
                      <span className="cur-page">{curPage}</span>
                      {isLastpage || isLoading ? (
                        <span className="item next disabled">
                          <PAGE_NEXT_DISABLED color="disabled" />
                        </span>
                      ) : (
                        <span className="item next">
                          <PAGE_NEXT />
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="setting-icon">
                    {/* <img alt="settings" width="20" src={SETTING_ICON} /> */}
                  </div>
                </div>
              </div>
              <div className="data-list">
                {isLoading ? (
                  <Loading />
                ) : (
                  <div className="table-wrap">
                    <div className="table-header">
                      <div className="table-item check-item">&nbsp;</div>
                      <div className="table-item header-item">
                        {t("taskList.table.name")}
                      </div>
                      <div className="table-item type">
                        {t("taskList.table.taskType")}
                      </div>
                      <div className="table-item times">
                        {t("taskList.table.runTime")}
                      </div>
                      <div className="table-item status">
                        {t("taskList.table.status")}
                      </div>
                      <div className="table-item time">Update Time</div>
                      <div className="table-item time">Create Time</div>
                    </div>
                    {taskListData.map((element: any, index: any) => {
                      const rowClass = classNames({
                        "table-row": true,
                        active:
                          curSelectTask &&
                          curSelectTask.taskName === element.taskName,
                      });
                      return (
                        <div
                          onClick={(event) => {
                            clickTaskInfo(element, event);
                          }}
                          data-uuid={element.taskName}
                          key={index}
                          className={rowClass}
                        >
                          <div className="table-item check-item center">
                            <input
                              checked={
                                curSelectTask
                                  ? curSelectTask.taskName === element.taskName
                                  : false
                              }
                              onChange={(event) => {
                                changeRadioSelect(event);
                              }}
                              type="radio"
                              name="taskList"
                            />
                          </div>
                          <div className="table-item header-item">
                            {element.taskType && (
                              <Link
                                to={`/create/step2/${element.taskType.toLowerCase()}/${
                                  element.taskName
                                }`}
                              >
                                {element.taskName}
                              </Link>
                            )}
                            {!element.taskType && (
                              <span>{element.taskName}</span>
                            )}
                            {/* <Link
                              to={`/create/step2/${element.taskType.toLowerCase()}/${
                                element.taskName
                              }`}
                            >
                              {element.taskName}
                            </Link> */}
                          </div>
                          <div className="table-item type">
                            {element.taskType
                              ? TYPE_LIST_MAP[element.taskType][nameStr]
                              : "-"}
                          </div>
                          <div className="table-item times">
                            {element.taskRunningTime}
                          </div>
                          <div className="table-item status">
                            <div
                              className={
                                element.taskStatus
                                  ? TASK_STATUS_MAP[element.taskStatus].class +
                                    " status"
                                  : "status"
                              }
                            >
                              <span className="icon">
                                {element.taskStatus
                                  ? STATUS_ICON_MAP[element.taskStatus]
                                  : ""}
                              </span>
                              {element.taskStatus
                                ? TASK_STATUS_MAP[element.taskStatus]?.en_name
                                : ""}
                            </div>
                          </div>
                          <div className="table-item time">
                            {element.updateTime ? (
                              <Moment format="YYYY-MM-DD HH:mm">
                                {element.updateTime}
                              </Moment>
                            ) : (
                              "-"
                            )}
                          </div>
                          <div className="table-item time">
                            {element.createTime ? (
                              <Moment format="YYYY-MM-DD HH:mm">
                                {element.createTime}
                              </Moment>
                            ) : (
                              "-"
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="bottom">
          <Bottom />
        </div>
      </div>
    </div>
  );
};

export default List;
