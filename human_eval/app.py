import argparse
import json
import os
import random
import time
from collections import Counter

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

random.seed(42)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(os.getcwd(), "data", "evaluation.db")
print(app.config["SQLALCHEMY_DATABASE_URI"])
app.config["SECRET_KEY"] = "123456"  # replace with a real secret key

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# GLOBAL VARIABLE for the comparison instances
COMPARISON_INSTANCES = []


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))


class EvaluationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    instance_index = db.Column(db.Integer)
    instance_id = db.Column(db.String(200))
    prompt = db.Column(db.String(1e4))
    model_a = db.Column(db.String(200))
    model_b = db.Column(db.String(200))
    completion_a = db.Column(db.String(1e4))
    completion_b = db.Column(db.String(1e4))
    completion_a_is_acceptable = db.Column(db.String(50))
    completion_b_is_acceptable = db.Column(db.String(50))
    preference = db.Column(db.String(50))
    instance_quality = db.Column(db.String(50))
    comment = db.Column(db.String(1e4))
    evaluator = db.Column(db.String(100))
    timestamp = db.Column(db.String(100))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))
        else:
            return "Invalid username or password"
    else:
        return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    else:
        return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
def index():
    # check if the user is logged in
    if current_user.is_authenticated:
        return redirect(url_for("instances", index=0, current_user=current_user))
    else:
        return redirect(url_for("login"))


@app.route("/instances/<int:index>")
def instances(index):
    return render_template("index.html", index=index, current_user=current_user)


@app.route("/api/model-outputs/<int:index>", methods=["GET"])
def get_model_outputs(index):
    if 0 <= index < len(COMPARISON_INSTANCES):
        prompt = COMPARISON_INSTANCES[index]["prompt"]
        completions = COMPARISON_INSTANCES[index]["completions"]
        random.shuffle(completions)
        return jsonify({"prompt": prompt, "completions": completions}), 200
    return jsonify({"error": "Index out of range"}), 200


@app.route("/summary", methods=["GET"])
@login_required
def summary():
    results = summarize_results()
    return jsonify(results), 200


def count_user_contributions(users, records):
    user_contributions = {}
    for user in users:
        user_contributions[user.username] = 0
    for record in records:
        user_contributions[record.evaluator] += 1
    user_contributions["all"] = len(records)
    return user_contributions


def get_progress(records):
    completed_instance_indices = set([record.instance_index for record in records])
    missing_instances = []
    for index in range(len(COMPARISON_INSTANCES)):
        if index not in completed_instance_indices:
            missing_instances.append(index)
    return {
        "completed": len(completed_instance_indices),
        "total": len(COMPARISON_INSTANCES),
        "missing_indices": missing_instances,
    }


def get_acceptance_results(records, target_model_a, target_model_b):
    acceptance_results = {target_model_a: {}, target_model_b: {}}
    for record in records:
        instance_id = record.instance_id
        if instance_id not in acceptance_results[record.model_a]:
            acceptance_results[record.model_a][instance_id] = []
        acceptance_results[record.model_a][instance_id].append(record.completion_a_is_acceptable)

        if instance_id not in acceptance_results[record.model_b]:
            acceptance_results[record.model_b][instance_id] = []
        acceptance_results[record.model_b][instance_id].append(record.completion_b_is_acceptable)

    # count how many instances get multiple annotations
    instances_with_multiple_annotations = [
        instance_id for instance_id, results in acceptance_results[record.model_a].items() if len(results) > 1
    ]
    agreement_results = {
        "num_instances_with_multiple_annotations": len(instances_with_multiple_annotations),
        "acceptance_agreement": None,
    }
    assert target_model_a in acceptance_results
    assert target_model_b in acceptance_results
    # get agreement on acceptance
    if len(instances_with_multiple_annotations) > 0:
        agreed_model_a_acceptance = 0
        agreed_model_b_acceptance = 0
        for instance_id in instances_with_multiple_annotations:
            if len(set(acceptance_results[target_model_a][instance_id][-2:])) == 1:
                agreed_model_a_acceptance += 1
            if len(set(acceptance_results[target_model_b][instance_id][-2:])) == 1:
                agreed_model_b_acceptance += 1
        agreement_results["acceptance_agreement"] = (agreed_model_a_acceptance + agreed_model_b_acceptance) / (
            2 * len(instances_with_multiple_annotations)
        )
        agreement_results[f"{target_model_a}_acceptance_agreement"] = agreed_model_a_acceptance / len(
            instances_with_multiple_annotations
        )
        agreement_results[f"{target_model_b}_acceptance_agreement"] = agreed_model_b_acceptance / len(
            instances_with_multiple_annotations
        )

    return {
        f"{target_model_a}": sum([1 if x[-1] == "yes" else 0 for _, x in acceptance_results[target_model_a].items()])
        / len(acceptance_results[target_model_a]),
        f"{target_model_b}": sum([1 if x[-1] == "yes" else 0 for _, x in acceptance_results[target_model_b].items()])
        / len(acceptance_results[target_model_b]),
        "agreement": agreement_results,
    }


def get_comparison_results(records, target_model_a, target_model_b):
    comparison_results = {}
    for record in records:
        instance_id = record.instance_id
        model_a = record.model_a
        model_b = record.model_b
        if instance_id not in comparison_results:
            comparison_results[instance_id] = []

        if record.preference == "a-is-better":
            comparison_results[instance_id].append(f"{model_a} is clearly better")
        elif record.preference == "a-is-slightly-better":
            comparison_results[instance_id].append(f"{model_a} is slightly better")
        elif record.preference == "b-is-better":
            comparison_results[instance_id].append(f"{model_b} is clearly better")
        elif record.preference == "b-is-slightly-better":
            comparison_results[instance_id].append(f"{model_b} is slightly better")
        elif record.preference == "tie":
            comparison_results[instance_id].append("tie")
        else:
            print("-------------------------------------")
            print("Unknown preference value.")
            print(record)

    # thre can be multiple annotations for each instance; use the latest comparison result for each instance
    latest_comparison_results = [results[-1] for _, results in comparison_results.items()]
    model_wins_counter = Counter(latest_comparison_results)
    model_wins_rates = {result: count / len(latest_comparison_results) for result, count in model_wins_counter.items()}
    # merge the clearly better and slightly better results
    model_wins_rates[f"{target_model_a}_wins"] = sum([v for k, v in model_wins_rates.items() if target_model_a in k])
    model_wins_rates[f"{target_model_b}_wins"] = sum([v for k, v in model_wins_rates.items() if target_model_b in k])

    # count how many instances get multiple annotations
    instances_with_multiple_annotations = [
        instance_id for instance_id, results in comparison_results.items() if len(results) > 1
    ]
    agreement_results = {
        "num_instances_with_multiple_annotations": len(instances_with_multiple_annotations),
        "comparison_agreement": None,
        "relexed_comparison_agreement": None,
    }
    if instances_with_multiple_annotations:
        agreed_comparison = 0
        relexed_agreed_comparison = 0
        for instance_id in instances_with_multiple_annotations:
            simplified_comparisons = []
            for comparison_result in comparison_results[instance_id]:
                if comparison_result == "tie":
                    simplified_comparisons.append("tie")
                elif target_model_a in comparison_result:
                    simplified_comparisons.append(target_model_a)
                elif target_model_b in comparison_result:
                    simplified_comparisons.append(target_model_b)
                else:
                    print("Unknown comparison result.")
                    print(comparison_result)
            if len(set(simplified_comparisons[-2:])) == 1:
                agreed_comparison += 1
                relexed_agreed_comparison += 1
            else:
                if "tie" in simplified_comparisons[-2:]:
                    relexed_agreed_comparison += 0.5
        agreement_results["comparison_agreement"] = agreed_comparison / len(instances_with_multiple_annotations)
        agreement_results["relexed_comparison_agreement"] = relexed_agreed_comparison / len(
            instances_with_multiple_annotations
        )

    model_wins_rates["agreement"] = agreement_results
    return model_wins_rates


def summarize_results():
    results = {}
    users = User.query.all()
    records = EvaluationRecord.query.all()

    # get the number of completed instances for all and each user
    results["user_contributions"] = count_user_contributions(users, records)

    # get the missing instances
    results["progress"] = get_progress(records)

    # get the comparison model pairs
    model_pairs = set([tuple(sorted([record.model_a, record.model_b])) for record in records])
    results["model_pairs"] = list(model_pairs)

    results["results"] = {}
    for target_model_a, target_model_b in model_pairs:
        feedback_records = {}
        comparison_records = []
        for record in records:
            # instance id is used to identify the comparison instance
            # there could be multiple records for the same instance
            instance_id = record.instance_id

            # skip if the record is not for the target model pair
            if set([target_model_a, target_model_b]) != set([record.model_a, record.model_b]):
                assert any([set([record.model_a, record.model_b]) == set(pair) for pair in model_pairs])
                continue

            # skip if the record is a feedback
            if record.instance_quality:
                if record.instance_quality not in feedback_records:
                    feedback_records[record.instance_quality] = []
                feedback_records[record.instance_quality].append(record.instance_index)
                continue

            comparison_records.append(record)

        acceptance_results = get_acceptance_results(comparison_records, target_model_a, target_model_b)
        comparison_results = get_comparison_results(comparison_records, target_model_a, target_model_b)
        results["results"][f"{target_model_a}_vs_{target_model_b}"] = {
            "acceptance_results": acceptance_results,
            "comparison_results": comparison_results,
            "feedback_records": feedback_records,
        }
    return results


@app.route("/api/submit-evaluation", methods=["POST"])
@login_required
def submit_evaluation():
    evaluation_data = request.get_json()
    print("Got new evaluation data:")
    print(evaluation_data)
    # write to the database
    new_record = EvaluationRecord(
        instance_index=evaluation_data["index"],
        instance_id=COMPARISON_INSTANCES[evaluation_data["index"]]["id"],
        prompt=evaluation_data["prompt"],
        model_a=evaluation_data["model_a"],
        model_b=evaluation_data["model_b"],
        completion_a=evaluation_data["completion_a"],
        completion_b=evaluation_data["completion_b"],
        completion_a_is_acceptable=evaluation_data["completion_a_is_acceptable"],
        completion_b_is_acceptable=evaluation_data["completion_b_is_acceptable"],
        preference=evaluation_data["preference"],
        instance_quality="",
        comment="",
        evaluator=evaluation_data["evaluator"],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"message": "Evaluation data submitted successfully"}), 200


@app.route("/api/submit-feedback", methods=["POST"])
@login_required
def submit_feedback():
    feedback_data = request.get_json()
    print("Got new feedback:")
    print(feedback_data)
    # write to the database
    new_record = EvaluationRecord(
        instance_index=feedback_data["index"],
        instance_id=COMPARISON_INSTANCES[feedback_data["index"]]["id"],
        prompt=feedback_data["prompt"],
        model_a=feedback_data["model_a"],
        model_b=feedback_data["model_b"],
        completion_a=feedback_data["completion_a"],
        completion_b=feedback_data["completion_b"],
        completion_a_is_acceptable="",
        completion_b_is_acceptable="",
        preference="",
        instance_quality=feedback_data["instance_quality"],
        comment=feedback_data["comment"],
        evaluator=feedback_data["evaluator"],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"message": "Evaluation data submitted successfully"}), 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison_data_path",
        type=str,
        required=True,
        help="The path to the data file containing the instances to be evaluated. "
        "Each instance should have a prompt and two completions.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="The host of the server.")
    parser.add_argument("--port", type=int, default=5001, help="The port of the server.")
    parser.add_argument("--debug", action="store_true", help="Whether to run the server in debug mode.")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(os.getcwd(), "data", "evaluation.db")):
        with app.app_context():
            db.create_all()
            new_user = User(username="admin", password=generate_password_hash("admin"))
            db.session.add(new_user)
            db.session.commit()

    # load the predictions
    global COMPARISON_INSTANCES
    with open(args.comparison_data_path) as f:
        COMPARISON_INSTANCES = [json.loads(line.strip()) for line in f.readlines()]

    print(f"Total number of comparison instances: {len(COMPARISON_INSTANCES)}")

    # run the app and listen on port 5000
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
