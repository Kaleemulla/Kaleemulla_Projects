import datetime
from dateutil import parser
import sys
import requests
import json
import yaml
import os
import re
import requests.adapters
import logging
from logging.handlers import RotatingFileHandler
from jira import JIRA
from flask import Flask
from flask import request
from flask_basicauth import BasicAuth

hello_options = {"", "hi", "hello", "?", "??", "help", "ask help", "options", "jedi", "jj"}

jedi_options = "You can ask me to perform the following actions:\n\n" \
               "\t1. Fetch/Get [Issue]\n" \
               "\t2. Fetch/Get [Field] [Issue]\n" \
               "\t3. Create Tasks\n" \
               "\t4. Log Work / Log / Log ip\n" \
               "\t5. Add Comment / Add\n" \
               "\t6. Close Task / Close\n" \
               "\t7. Get Summary / Sum\n"

ack_log_work = "Logged work successfully"
ack_log_work_close = "Thread closed. Logged work and closed task successfully"
ack_close_task = "Closed task successfully and set remaining estimates to 0"
ack_comment = "Added comment successfully"
ack_create_tasks = "Created sub-task successfully and added original estimate"

message_tags = {"create_task": "[Create Task]:\n", "log_work": "[Log Work]:\n", "add_comment": "[Add Comment]:\n",
                "close_task": "[Close Task]:\n"}

jedi_fetch_stories_keywords = ["fetch my stories", "fetch all stories", "fetch stories"]
jedi_fetch_fields = ["fetch", "get", "show", "search", "list"]
# jedi_log_keywords = ["log time", "log work", "log hr", "log hrs", "log task", "add hr", "add hrs", "add work", "close task"]
jedi_log_keywords = ["log time", "log work", "log hr", "log hrs", "log task", "add hr", "add hrs", "add work"]
jedi_log_ip_keywords = ["log ip", "log time ip", "log work ip", "log hr ip", "log hrs ip", "log task ip", "add hr ip", "add hrs ip", "add work ip"]
jedi_comment_keywords = ["add comment", "create comment"]
jedi_add_desc_keywords = ["add desc", "add descr", "add description"]
jedi_get_summary_keywords = ["summary", "get summary", "get my summary", "fetch summary", "fetch my summary"]
jedi_close_task_keywords = ["close", "done"]
# jedi_get_sprint_summary_keywords = ["get sprint summary", "get sprint sum", "get all summary", "get all sum"]
# jedi_get_engineering_summary = ["get engineering summary", "get eng summary", "get eng sum"]
jedi_configure_keywords = ["configure", "add config"]
jedi_create_task_keywords = ["add task", "create task"]

p_jedi_fetch_keywords = re.compile('|'.join(map(re.escape, jedi_fetch_fields)), re.IGNORECASE)
p_jedi_configure_keywords = re.compile('|'.join(map(re.escape, jedi_configure_keywords)), re.IGNORECASE)
p_jedi_log_keywords = re.compile('|'.join(map(re.escape, jedi_log_keywords)), re.IGNORECASE)
p_jedi_log_ip_keywords = re.compile('|'.join(map(re.escape, jedi_log_ip_keywords)), re.IGNORECASE)
p_jedi_close_task_keywords = re.compile('|'.join(map(re.escape, jedi_close_task_keywords)), re.IGNORECASE)
p_jedi_comment_keywords = re.compile('|'.join(map(re.escape, jedi_comment_keywords)), re.IGNORECASE)
p_jedi_add_desc_keywords = re.compile('|'.join(map(re.escape, jedi_add_desc_keywords)), re.IGNORECASE)
p_jedi_get_summary_keywords = re.compile('|'.join(map(re.escape, jedi_get_summary_keywords)), re.IGNORECASE)
p_jedi_create_task_keywords = re.compile('|'.join(map(re.escape, jedi_create_task_keywords)), re.IGNORECASE)

with open("config.yml", 'r') as yml_file:
    cfg = yaml.load(yml_file, Loader=yaml.Loader)

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = cfg["rp_username"]
app.config['BASIC_AUTH_PASSWORD'] = cfg["rp_password"]

basic_auth = BasicAuth(app)
app.config['BASIC_AUTH_FORCE'] = True

directory = f"{os.getcwd()}/message_api_key"

if not os.path.exists(directory):
    os.makedirs(directory)

'''def initialize_llm():
    print("Initilize llm")
    llm = ChatOllama(model="llama3:8b", temperature=0.5)
    return llm'''


def initialize_jira(jira_user, jira_api_key):
    jira = JIRA(jira_url, basic_auth=(jira_user, jira_api_key))
    return jira


def get_current_sprint(issues):
    """
    Fetches the current active sprint from the first issue with a valid custom field.

    Args:
        issues: A list of JIRA issue objects.

    Returns:
        The name of the current active sprint, or None if no sprint found.
    """
    for issue in issues:
        if issue.fields.customfield_10020:
            for sprint in issue.fields.customfield_10020:
                if sprint.name:
                    match = re.search(r'S(\d+)', sprint.name)
                    if match:
                        return sprint.name
            break  # Break after processing the first issue

    return None


def get_tasks(jira, member_id, ip=False):
    project = "CMS"
    jql_query = f"project = {project} AND type = 'Sub-task' AND sprint in openSprints() AND assignee = '{member_id}'"
    if ip:
        jql_query += " AND status = 'In Progress'"
    # jql_query = "issue=CMS-54558"
    tasks_in_current_sprint = jira.search_issues(jql_query)
    return tasks_in_current_sprint if tasks_in_current_sprint else None


def get_stories_user(jira, member_id):
    project = "CMS"
    jql_query = f"project = {project} AND type = 'Story' AND assignee = '{member_id}'"
    # jql_query = "issue=CMS-49526"
    tasks_in_current_sprint = jira.search_issues(jql_query)
    return tasks_in_current_sprint if tasks_in_current_sprint else None


def create_sub_task(jira, issue, description, estimate):
    project = "CMS"
    issue_dict = {
        'project': {'key': project},
        'summary': description,  # Need to parse it from thread message
        'description': description,
        'issuetype': {'name': 'Sub-task'},
        'parent': {'key': issue}
    }
    new_issue = jira.create_issue(fields=issue_dict)
    new_issue.update(fields={
        'timetracking': {
            'originalEstimate': estimate  # from hours to JIRA's expected format
        }
    })
    return new_issue if new_issue else None


def get_stories(jira):
    project = "CMS"
    jql_query = f"project = {project} AND (type='Story' OR type='Bug')"
    # jql_query = "issue=CMS-54797"
    stories_in_current_sprint = jira.search_issues(jql_query)
    return stories_in_current_sprint if stories_in_current_sprint else None


def get_current_sprint_end_date(issues):
    for issue in issues:
        if issue.fields.customfield_10020:
            for sprint in issue.fields.customfield_10020:
                if sprint.endDate:
                    # Parse the ISO 8601 formatted date string
                    try:
                        end_date = datetime.datetime.fromisoformat(sprint.endDate).date().strftime('%Y-%m-%d')
                        return end_date
                    except ValueError:
                        print(f"Error parsing date: {sprint.endDate}")
        break  # Break after processing the first issue

    return None


def get_next_sprint_details(jira, member_id, message_date):
    project = "CMS"
    jql_query = f"project = {project} AND type = 'Sub-task' AND sprint in closedSprints() AND assignee = '{member_id}'"
    issues = jira.search_issues(jql_query)
    sprint_name = get_current_sprint(issues)
    print(sprint_name)
    num = int(re.findall(r'S(\d+)', sprint_name)[0]) + 1
    next_sprint_name = "FY25Q2 CPRTB S" + str(num)
    # print(next_sprint_name)
    jql_query = f"project = {project} AND type = 'Sub-task' AND sprint = '{next_sprint_name}' AND assignee = '{member_id}'"
    nextissues = jira.search_issues(jql_query)
    print(nextissues)
    # current_date = datetime.datetime.fromisoformat(message_date).date().strftime('%Y-%m-%d')
    current_date = parser.isoparse(message_date).strftime('%Y-%m-%d')
    print(current_date)
    print(get_current_sprint_end_date(issues))
    return get_current_sprint_end_date(issues) == current_date, nextissues


def close_task(jira, issue_key):
    jira.add_comment(issue_key, "Closing this issue, fulfilled the requirements")
    issue = jira.issue(issue_key)
    print(issue.fields.status)
    original_estimate = issue.fields.timetracking.originalEstimateSeconds /3600
    if issue.fields.status != "Done":
        remaining_estimate = 0
        jira.transition_issue(issue, "Done")
        issue.update(fields={
            'timetracking': {
                'originalEstimate': str(original_estimate) + 'h',
                'remainingEstimate': str(remaining_estimate) + 'h'  # from hours to JIRA expected format
            }
        })


def get_next_sprint_summary(tasks, member_id):
    totalTimeSpent = 0
    totalOriginalEstimate = 0
    issueCount = 0
    totalRemainingEstimate = 0
    totalWorkRatio = 0
    current_sprint = ""
    if tasks:
        current_sprint = get_current_sprint(tasks)
        for task in tasks:
            if task.fields.timeoriginalestimate:
                totalOriginalEstimate += task.fields.timeoriginalestimate
            if task.fields.timespent:
                totalTimeSpent += task.fields.timespent
            if task.fields.timetracking.remainingEstimateSeconds:
                totalRemainingEstimate += task.fields.timetracking.remainingEstimateSeconds
            if task.fields.workratio and task.fields.workratio != -1 and task.fields.workratio != 1:
                totalWorkRatio += task.fields.workratio
                issueCount += 1
        summary = f"<b> Number of Tasks assigned:</b> {len(tasks)}<br>" \
                  f"<b>Currently working on :</b> {issueCount} issues<br>" \
                  f"<b>Sprint Name:</b> {current_sprint}<br>" \
                  f"<b>Total Time Spent:</b> {(totalTimeSpent / 3600):.2f} h<br>" \
                  f"<b>Total Original Estimate:</b> {(totalOriginalEstimate / 3600):.2f} h<br>" \
                  f"<b>Total Remaining Estimate:</b> {(totalRemainingEstimate / 3600):.1f} h<br>"
    else:
        summary = f"No issues for for user {member_id}"

    return summary


def get_tasks_summary(jira, member_id):
    totalTimeSpent = 0
    totalOriginalEstimate = 0
    issueCount = 0
    totalRemainingEstimate = 0
    totalWorkRatio = 0
    current_sprint = ""
    tasks = get_tasks(jira, member_id)
    if tasks:
        current_sprint = get_current_sprint(tasks)
        for task in tasks:
            if task.fields.timeoriginalestimate:
                totalOriginalEstimate += task.fields.timeoriginalestimate
            if task.fields.timespent:
                totalTimeSpent += task.fields.timespent
            if task.fields.timetracking.remainingEstimateSeconds:
                totalRemainingEstimate += task.fields.timetracking.remainingEstimateSeconds
            if task.fields.workratio and task.fields.workratio != -1 and task.fields.workratio != 1:
                totalWorkRatio += task.fields.workratio
                issueCount += 1

        # print(f"Number of Tasks assigned: {len(tasks)}")
        # print(f"Currently working on : {issueCount} issues")
        averageWorkRatio = totalWorkRatio / issueCount if issueCount > 0 else -1
        # print(f"Sprint Name: {current_sprint} ")
        # print(f"Total Time Spent: {totalTimeSpent / 3600} h")
        # print(f"Total Original Estimate: {totalOriginalEstimate / 3600} h")
        # print(f"Total Remaining Estimate: {totalRemainingEstimate / 3600} h")
        # if averageWorkRatio == -1:
        #    print("Average Work Ratio: -1")
        # else:
        #    print(f"Average Work Ratio: {averageWorkRatio} %")
        # print("-" * 20)
        summary = f"<b> Number of Tasks assigned:</b>{len(tasks)}<br>" \
                  f"<b>Currently working on:</b> {issueCount} issues<br>" \
                  f"<b>Sprint Name:</b> {current_sprint}<br>" \
                  f"<b>Total Time Spent:</b> {(totalTimeSpent / 3600):.2f} h<br>" \
                  f"<b>Total Original Estimate:</b> {(totalOriginalEstimate / 3600):.2f} h<br>" \
                  f"<b>Total Remaining Estimate:</b> {(totalRemainingEstimate / 3600):.1f} h<br>"
    else:
        summary = f"No issues for for user {member_id}"

    '''summary = "Number of Tasks assigned: 13" \
                   "\nCurrently working on : 3 issues" \
                   "\nSprint Name: FY25Q2 CPRTB S7" \
                   "\nTotal Time Spent: 17h" \
                   "\nTotal Original Estimate: 64h" \
                   "\nAverage Work Ratio: 100.0 %"'''

    return summary


def convert_to_hours(message):
    hours_per_day = 8
    total_hours = 0
    matches = re.findall(r'(\d+)([hmd])', message)

    for value, unit in matches:
        value = int(value)
        if unit == 'h':
            total_hours += value
        elif unit == 'm':
            total_hours += value / 60
        elif unit == 'd':
            total_hours += value * hours_per_day

    return total_hours


def update_task_and_estimate(jira, issue_key, hours_burnt):
    # issue_key = "CMS-54558"
    issue = jira.issue(issue_key)
    status = issue.fields.status.name

    if status == "To Do":
        jira.transition_issue(issue, "In Progress")

    #attrs = vars(issue)
    #print(', '.join("%s: %s" % item for item in attrs.items()))
    original_estimate = issue.fields.timetracking.originalEstimateSeconds /3600
    if hasattr(issue.fields.timetracking, 'timeSpentSeconds'):
        time_spent = issue.fields.timetracking.timeSpentSeconds /3600
    else:
        time_spent = 0
    new_remaining_estimate = original_estimate - time_spent - hours_burnt

    # Number of hours burnt is more than original and task is in progress
    if new_remaining_estimate < 0:
        if status == "In Progress":
            new_remaining_estimate = 1
            print("setting new estimate to 1")
        else:
            new_remaining_estimate = 0

    jira.add_worklog(issue_key, timeSpent=str(hours_burnt) + 'h')
    issue.update(fields={
        'timetracking': {
            'originalEstimate': str(original_estimate) + 'h',
            'remainingEstimate': str(new_remaining_estimate) + 'h'  # from hours to JIRA's expected format
        }
    })

    return f"Updated issue {issue_key} with {hours_burnt} hours burnt. New remaining estimate: {new_remaining_estimate} hours"


def fetch_jira_issue(jira, issue_key):
    issue = jira.issue(issue_key)
    attrs = vars(issue)
    return attrs


def check_and_notify(jira):
    # project = "CMS"
    # jql_query = f"project = '{project}' AND status IN ('In Progress', 'To Do') AND timeestimate > 0"
    # issues = jira.search_issues(jql_query)

    issue = jira.issue("CMS-54558")
    # for issue in issues:
    # assignee_email = issue.fields.assignee.emailAddress
    body = "Action Required: Task {} - {}\n"
    body = f"Hi {issue.fields.assignee.displayName},\n\nPlease take action on the following task:\n\n"
    body += f"Task ID: {issue.key}\n"
    body += f"Summary: {issue.fields.summary}\n"
    body += f"Remaining Time Estimate: {issue.fields.timeestimate / 3600}h\n"
    # body += f"Sprint End Date: {"2024 - 11 - 19"}\n\n"
    body += "Please ensure the task is closed or the time estimate is updated before the sprint end date.\n\n"
    body += "Best regards,\n"
    body += "Jira Jedi"

    return body


def send_get(url, payload=None, js=True):
    if payload == None:
        request = requests.get(url, headers=headers)
    else:
        request = requests.get(url, headers=headers, params=payload)
    if js == True:
        request = request.json()
    return request


def send_post(url, data):
    request = requests.post(url, json.dumps(data), headers=headers).json()
    return request


def post_message(room_id, message, parentId=None):
    logger.debug("Posting message to " + room_id)
    logger.debug(message)

    payload = {"roomId": room_id, "html": message}

    if parentId:
        payload.update({"parentId": parentId})

    return requests.post(f"{spark_url}/v1/messages", json.dumps(payload), headers=headers).json()


def post_message_markdown(room_id, message):
    logger.debug("Posting message to " + room_id)
    logger.debug(message)
    payload = {"roomId": room_id, "markdown": message}

    return requests.post(f"{spark_url}/v1/messages", json.dumps(payload), headers=headers).json()


def post_message_to_bot_room(message):
    post_message(cfg["bot_LoggingRoomId"], message)


def init_logger():
    logger = logging.getLogger("botlog")
    logging.basicConfig(
        handlers=[
            RotatingFileHandler(cfg["logfile_name"], maxBytes=cfg["logfile_size"], backupCount=cfg["logfile_count"])],
        level=cfg["log_level"],
        format='%(asctime)s %(levelname)-8s %(thread)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger.debug("Initializing logger")
    return logger


@basic_auth.required
@app.route('/', methods=['GET', 'POST'])
def jira_jedi_webhook():
    print(request)
    if request.method == 'GET':
        return "Success"

    webhook = request.get_json(silent=True)
    room_id = webhook["data"]["roomId"]
    print(webhook)

    if webhook["resource"] == "room" and webhook["event"] == "created":
        post_message_markdown(room_id,
                              "<@all><BR>Welcome to a new room with Jira Jedi!! You have to call me specifically with `@%s` to get response." % bot_name)
        return "true"

    if webhook["resource"] == "memberships" and webhook["data"]["roomType"] == "group":
        post_message_markdown(room_id,
                              "<@all><BR>Hello from Jira Jedi!! As this is a group room, you have to call me specifically with `@%s` to get response." % bot_name)
        return "true"

    email = webhook["data"]["personEmail"]

    if webhook["resource"] == "memberships":
        if email == bot_email:
            msg = "Hi there. Type **help** to see available options."
        else:
            msg = "Welcome <@personEmail:{}> to the group room. If you want me to answer, you need to call me specifically with @<@personId:{}>".format(
                email, cfg["bot_id"])

        post_message_markdown(room_id, msg)
        return "true"

    if "@webex.bot" not in email:
        try:
            message_id = webhook['data']['id']
            member_id = webhook['data']['personEmail']
            message_created = webhook['data']['created']
            jira_user = webhook['data']['personEmail']
            jira_api_key = str()

            result = send_get(f'{spark_url}/v1/messages/{message_id}')
            message = result.get('text', '')
            print(f'Raw message: {message}')

            for name in bot_name.split(" "):
                message = message.replace(name + " ", '')
            message = message.replace(bot_name + " ", '')

            print(f'Processing message: {message}')

            if 'my_api_key' in message.lower():
                fp = open(f"message_api_key/{jira_user}", "w+")
                fp.write(message_id)
                fp.close()

                return '', 200

            if not os.path.exists(f"message_api_key/{jira_user}"):
                logger.info(f"message_api_key/{jira_user} file does not exists")
                send_message = f"Please set your <b>JIRA_API_Key</b> as a message in this space with format <b>'my_api_key: xxx'</b>"
                post_message(room_id, send_message)
                return '', 200

            fp = open(f"message_api_key/{jira_user}", "r")
            api_key_message_id = fp.read()
            fp.close()

            if not api_key_message_id:
                logger.info("api_key_message_id is empty")
                send_message = f"Please set your <b>JIRA_API_Key</b> as a message in this space with format <b>'my_api_key: xxx'</b>"
                post_message(room_id, send_message)
                return '', 200

            try:
                api_key_result = send_get(f'{spark_url}/v1/messages/{api_key_message_id}')
                api_key_message = api_key_result.get('text', '')
                jira_api_key = re.split(":\s*", api_key_message)[-1]
            except Exception as ae:
                logger.info(f"Exception fetching api_key_message: {ae}")
                send_message = f"Please set your <b>JIRA_API_Key</b> as a message in this space with format <b>'my_api_key: xxx'</b>"
                post_message(room_id, send_message)
                return '', 200

            if not jira_api_key:
                logger.info("jira_api_key is empty")
                send_message = f"Please set your <b>JIRA_API_Key</b> as a message in this space with format <b>'my_api_key: xxx'</b>"
                post_message(room_id, send_message)
                return '', 200

            jira = initialize_jira(jira_user, jira_api_key)

            if "parentId" not in webhook["data"]:  # This is a single message

                if message == "log" or message == "log ip" or p_jedi_log_keywords.search(message) or p_jedi_log_ip_keywords.search(message):
                    if "ip" == message[-2:]:
                        user_tasks = get_tasks(jira, member_id, True)
                    else:
                        user_tasks = get_tasks(jira, member_id)

                    for task in user_tasks:
                        if task.fields.status.name != "Done":
                            task_key = task.key
                            task_type = task.fields.issuetype
                            task_summary = task.fields.summary
                            parent_number = task.fields.parent.key
                            parent_summary = task.fields.parent.fields.summary
                            parent_type = task.fields.parent.fields.issuetype
                            send_message = f"{message_tags['log_work']} {parent_type}: {parent_number}: {parent_summary}\n {task_type}: {task_key} : {task_summary} [{task.fields.timetracking.remainingEstimateSeconds/3600}h/{task.fields.timeoriginalestimate/3600}h remaining] - {task.fields.status}"
                            post_message(room_id, send_message)
                    return '', 200

                elif message == "comment" or p_jedi_comment_keywords.search(message):
                    user_tasks = get_tasks(jira, member_id)

                    for task in user_tasks:
                        # if task.fields.status.name != "Done":
                        task_key = task.key
                        task_summary = task.fields.summary
                        task_type = task.fields.issuetype
                        parent_number = task.fields.parent.key
                        parent_summary = task.fields.parent.fields.summary
                        parent_type = task.fields.parent.fields.issuetype
                        message = f"{message_tags['add_comment']} {parent_type}: {parent_number}: {parent_summary}\n {task_type}: {task_key} : {task_summary} - {task.fields.status}"
                        send_message = message
                        post_message(room_id, send_message)
                    return '', 200

                elif message == "sum" or p_jedi_get_summary_keywords.search(message):
                    is_sprint_end_date, next_sprint_issues = get_next_sprint_details(jira, member_id, message_created)
                    print(is_sprint_end_date)
                    if is_sprint_end_date:
                        send_message = get_next_sprint_summary(next_sprint_issues, member_id)
                    else:
                        send_message = get_tasks_summary(jira, member_id)
                    post_message(room_id, send_message)
                    return '', 200

                elif message == "close" or p_jedi_close_task_keywords.search(message):
                    user_tasks = get_tasks(jira, member_id)

                    for task in user_tasks:
                        # if task.fields.status.name != "Done":
                        task_key = task.key
                        task_summary = task.fields.summary
                        task_type = task.fields.issuetype
                        parent_number = task.fields.parent.key
                        parent_summary = task.fields.parent.fields.summary
                        parent_type = task.fields.parent.fields.issuetype
                        message = f"{message_tags['close_task']} {parent_type}: {parent_number}: {parent_summary}\n {task_type}: {task_key} : {task_summary} - {task.fields.status}"
                        send_message = message
                        post_message(room_id, send_message)
                    return '', 200

                elif message == "create" or p_jedi_create_task_keywords.search(message):
                    issues_sprint = get_stories(jira)
                    print(issues_sprint)

                    for issue in issues_sprint:
                        key = issue.key
                        summary = issue.fields.summary
                        type = issue.fields.issuetype
                        message = f"{message_tags['create_task']} {type}: {key}: {summary} - {issue.fields.status}"
                        send_message = message
                        post_message(room_id, send_message)
                    return '', 200

                elif p_jedi_fetch_keywords.search(message):
                    try:
                        issue_key = re.findall('CMS-[0-9]*', message)[-1]
                        issue = jira.issue(issue_key)
                        send_message = str()

                        m_split = re.split("\s+", message)
                        print(m_split)
                        if len(m_split) == 2:
                            send_message = f"<b> Issue:</b> {m_split[1]}\n\n<b>Summary:</b> {issue.fields.summary}\n\n<b>Description:</b> {issue.fields.description}\n\n<b>Status:</b> {issue.fields.status}\n\n<b>Assignee:</b> {issue.fields.assignee}"
                        if len(m_split) == 3:
                            send_message = f"<b> Issue:</b> {m_split[2]}\n\n<b>{m_split[1]}:</b> {getattr(issue.fields,m_split[1])}"
                        
                        print(send_message)
                        post_message(room_id, send_message)
                        return '', 200
                    except Exception as te:
                        send_message = f"Exception while fetching issue: {ae}"
                        post_message(room_id, send_message)
                        return '', 200

                elif message.lower() in hello_options:
                    post_message(room_id, jedi_options)
                    
                    return '', 200

                else:
                    post_message(room_id,
                                 "Currently this option is not supported\n\nTry other options like summary, log, comment")
                    return '', 200

            # message_tags = {"create_task": "[Create Task]:\n", "log_work": "[Log Work]:\n", "add_comment": "[Add Comment]:\n", "close_task": "[Close Task]:\n"}

            if "parentId" in webhook["data"]:  # This is a thread message
                print(message, message_id)
                parent_id = webhook["data"]["parentId"]

                result = send_get(f'{spark_url}/v1/messages/{parent_id}')
                parent_message = result.get('text', '')
                parent_message = parent_message.replace(bot_name.lower() + " ", '')

                parent_message_tag = parent_message.splitlines()[0].strip()
                issue_key = re.findall('CMS-[0-9]*', parent_message)[-1]

                if parent_message_tag == message_tags["log_work"].strip():
                    # extract numeric part from message
                    hours_to_be_logged = convert_to_hours(message)
                    print(issue_key)
                    print(hours_to_be_logged)
                    if issue_key and hours_to_be_logged:
                        send_message = update_task_and_estimate(jira, issue_key, hours_to_be_logged)
                        # send_message = f"{ack_log_work} for {issue_key}"
                        post_message(room_id, send_message, parent_id)
                    return '', 200

                elif parent_message_tag == message_tags["add_comment"].strip():
                    # Perform action add comment
                    comment_message = f"@rorajpal: {message}"
                    jira.add_comment(issue_key, comment_message)
                    send_message = f"{ack_comment} for {issue_key}"
                    post_message(room_id, send_message, parent_id)
                    return '', 200

                elif parent_message_tag == message_tags["create_task"].strip():
                    # Perform action to create task comment
                    original_estimate = re.findall(r"[0-9]h", message)[0]
                    task_description = message.replace(original_estimate, '').strip()
                    print(original_estimate)
                    print(task_description)
                    new_issue = create_sub_task(jira, issue_key, task_description, original_estimate)
                    send_message = f"{ack_create_tasks} of {original_estimate} for {new_issue}"
                    post_message(room_id, send_message, parent_id)
                    return '', 200

                elif parent_message_tag == message_tags["close_task"].strip():
                    print(issue_key)
                    close_task(jira, issue_key)
                    send_message = f"{ack_close_task} for {issue_key}"
                    post_message(room_id, send_message, parent_id)
                    return '', 200

                else:
                    send_message = f"No action can be performed on this thread message"
                    post_message(room_id, send_message)
                    return '', 200

        except Exception as e:
            print(e)
            logger.error(e)
            return '', 500


if __name__ == "__main__":
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer " + cfg["bot_access_token"]
    }

    logger = init_logger()
    ip = "0.0.0.0"
    port = 7001

    spark_url = cfg["spark_url"]
    jira_url = cfg["jira_url"]


    if len(cfg["bot_access_token"]) != 0:
        test_auth = send_get(f"{spark_url}/v1/people/me", js=False)
        if test_auth.status_code == 401:
            print("Looks like the provided access token is not correct.\n")
            sys.exit()
        if test_auth.status_code == 200:
            test_auth = test_auth.json()
            bot_name = test_auth.get("displayName", "")
            bot_email = test_auth.get("emails", "")[0]

            if "@webex.bot" not in bot_email:
                logger.error("You have provided an access token which does not relate to a Bot Account.")
                print("You have provided an access token which does not relate to a Bot Account.\n")
                sys.exit()
    else:
        print("'bearer' variable is empty! \n")
        sys.exit()

    if os.path.exists(cfg["crt_file_path"]) and os.path.exists(cfg["key_file_path"]):
        logger.info("Started rally bot on https port " + str(port))
        app.run(host=ip, port=port, ssl_context=(cfg["crt_file_path"], cfg["key_file_path"]))
    else:
        logger.info("Started rally bot on http port " + str(port))
        app.run(host=ip, port=port, ssl_context=('/home/cmsbuild/jira_jedi/cert.pem', '/home/cmsbuild/jira_jedi/key.pem'))
