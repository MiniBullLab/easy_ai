import { Action, IState } from "./Store";
import { ACTION_TYPES } from "./types";

export default function reducer(
  state: IState | null | undefined,
  action: Action
) {
  if (!state) {
    return null;
  }

  switch (action.type) {
    case ACTION_TYPES.OPEN_SIDE_BAR: {
      return {
        ...state,
        isOpen: true,
      };
    }
    case ACTION_TYPES.CLOSE_SIDE_BAR: {
      return {
        ...state,
        isOpen: false,
      };
    }
    case ACTION_TYPES.SET_API_URL: {
      return {
        ...state,
        apiUrl: action.apiUrl,
      };
    }
    case ACTION_TYPES.OPEN_INFO_BAR: {
      return {
        ...state,
        infoIsOpen: true,
      };
    }
    case ACTION_TYPES.CLOSE_INFO_BAR: {
      return {
        ...state,
        infoIsOpen: false,
      };
    }
    case ACTION_TYPES.SET_S3_IMPORT: {
      return {
        ...state,
        s3IsImporting: action.s3IsImporting,
      };
    }
    case ACTION_TYPES.UPDATE_TASK_INFO: {
      return {
        ...state,
        tmpTaskInfo: action.taskInfo,
      };
    }
    default:
      return state;
  }
}
