import { createStore } from "redux";
import reducer from "./Reducer";
import { ACTION_TYPES } from "./types";

export interface IState {
  apiUrl: string;
  createTaskFlag: boolean;
  tmpTaskInfo: any;
  infoIsOpen?: boolean;
  isOpen: boolean;
  s3IsImporting: boolean;
  lastUpdated: number;
}

export type Action =
  | {
      type: ACTION_TYPES.OPEN_SIDE_BAR;
    }
  | {
      type: ACTION_TYPES.CLOSE_SIDE_BAR;
    }
  | {
      type: ACTION_TYPES.OPEN_INFO_BAR;
    }
  | {
      type: ACTION_TYPES.CLOSE_INFO_BAR;
    }
  | {
      type: ACTION_TYPES.SET_S3_IMPORT;
      s3IsImporting: boolean;
    }
  | {
      type: ACTION_TYPES.SET_API_URL;
      apiUrl: string;
    }
  | {
      type: ACTION_TYPES.UPDATE_TASK_INFO;
      taskInfo: any;
    };

export function makeStore(): any {
  return createStore(reducer, {
    apiUrl: "",
    createTaskFlag: false,
    tmpTaskInfo: {},
    infoIsOpen: false,
    isOpen: false,
    s3IsImporting: false,
    // isOpen: localStorage.getItem("drhIsOpen") ? true : false,
    lastUpdated: 0,
  });
}
