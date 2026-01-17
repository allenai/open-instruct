// Global variable to store the current index
let current_index = instance_index;

// Fetch the initial model outputs based on the instance index
rendere_instance(current_index);

// Fetch the model outputs from the API and update the UI
async function rendere_instance(index) {
    const response = await fetch(`/api/model-outputs/${index}`);
    const data = await response.json();

    // if the response is error, show the out of range message
    if (data.error == "Index out of range") {
        show_alert(
            "You requested an out-of-range instance. You might have completed all the evaluations. Thank you for your contribution!",
            "danger",
            insert_after_selector="#instance-info",
            timeout=1e10 // set timeout to a very large number so that the alert doesn't disappear
        );
        clear_all();
        return;
    }

    clear_all();
    $("#instance-id").html(`Instance ${index}`);

    // let's use a unified format here that support multiple messages, though currently we only have one user prompt.
    var messages = [{"role": "user", "text": data.prompt}];
    var history_message_region = $("#history-message-region");
    history_message_region.empty();

    $.each(messages, function(i, message) {
        var icon = message.role == "user" ? "🧑" : "🤖";

        var $message_element = $("<div></div>").addClass("row").html(`
            <div class="col icon-col">
                <button class="role-icon">${icon}</button>
            </div>
            <div class="col message-col history-message-col">
                <xmp class="message-text">${message.text}</xmp>
            </div>
        `);

        history_message_region.append($message_element);
    });

    // now render the completions
    completion_a = data.completions[0];
    completion_b = data.completions[1];

    $("#completion-A-col").html(`
        <xmp class="message-text" id="${completion_a.model}-completion">${completion_a.completion}</xmp>
    `);
    $("#completion-B-col").html(`
        <xmp class="message-text" id="${completion_b.model}-completion">${completion_b.completion}</xmp>
    `);

    // Change the URL path with the current index
    window.history.pushState(null, '', `/instances/${index}`);
}


// clear everything
function clear_all() {
    $('#history-message-region').html(`
        <div class="row">
            <div class="col icon-col">
                <button class="role-icon">🧑</button>
            </div>
            <div class="col message-col history-message-col">
                <xmp class="message-text"></xmp>
            </div>
        </div>
    `);
    $('.completion-col').empty();
    $('input[type="checkbox"], input[type="radio"]').prop('checked', false);
    $('textarea').val('');
}


function show_alert(message, type, insert_after_selector, timeout=5000) {
    const alert_container = $(`<div class="alert alert-${type} mx-auto mt-2" style="max-width:500px" role="alert">${message}</div>`)[0];
    $(insert_after_selector)[0].insertAdjacentElement("afterend", alert_container);
    setTimeout(() => {
        alert_container.remove();
    }, timeout);
}

async function submit_evaluation() {
    try {
        // get the model name by trimming out the last `-completion` part
        const model_a = $("#completion-A-col").find("xmp").attr("id").slice(0, -11);
        const model_b = $("#completion-B-col").find("xmp").attr("id").slice(0, -11);
        const completion_a_is_acceptable = $("input[name='a-is-acceptable']:checked").val();
        const completion_b_is_acceptable = $("input[name='b-is-acceptable']:checked").val();
        const preference = $("input[name='preference-selection']:checked").val();

        // get the prompt and completions
        const prompt = $("#history-message-region").find("xmp").text();
        const completion_a = $("#completion-A-col").find("xmp").text();
        const completion_b = $("#completion-B-col").find("xmp").text();

        // make sure all the required fields are filled
        if (completion_a_is_acceptable == undefined || completion_b_is_acceptable == undefined || preference == undefined) {
            show_alert("Please fill in all the questions.", "danger", insert_after_selector="#evaluation-submit", timeout=5000);
            return;
        }
        const response = await fetch("/api/submit-evaluation", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                index: current_index,
                model_a,
                model_b,
                prompt,
                completion_a,
                completion_b,
                completion_a_is_acceptable,
                completion_b_is_acceptable,
                preference,
                evaluator: username
            }),
        });

        // if the response is 200, show the success message
        if (response.status == 200) {
            show_alert("Evaluation data is submitted successfully.", "success", insert_after_selector="#evaluation-submit", timeoutput=5000);
            console.log("Evaluation data is submitted successfully.");
            current_index++;
            rendere_instance(current_index);
        }
        else if (response.status == 401) {
            show_alert("You need to log in to submit evaluation data.", "danger", insert_after_selector="#evaluation-submit", timeoutput=5000);
        }
        else {
            console.log(response);
            show_alert("Error when submitting evaluation data. Please try again.", "danger", insert_after_selector="#evaluation-submit", timeoutput=5000);
            console.error("Error when submitting evaluation data:", response.status);
        }
    } catch (error) {
        show_alert("Error when submitting evaluation data. Please try again.", "danger", insert_after_selector="#evaluation-submit", timeoutput=5000);
        console.error("Error when submitting evaluation data:", error);
    }
}

$("#evaluation-submit").click(function () {
    // prevent default form submission
    event.preventDefault();
    submit_evaluation();
});



async function submit_feedback() {
    try {
        // get the model name by trimming out the last `-completion` part
        const model_a = $("#completion-A-col").find("xmp").attr("id").slice(0, -11);
        const model_b = $("#completion-B-col").find("xmp").attr("id").slice(0, -11);

        // get the prompt and completions
        const prompt = $("#history-message-region").find("xmp").text();
        const completion_a = $("#completion-A-col").find("xmp").text();
        const completion_b = $("#completion-B-col").find("xmp").text();

        // feedback
        const instance_quality = $("input[name='instance-quality']:checked").val();
        const comment = $("textarea[name='comment']").val();

        console.log("instance_quality:", instance_quality);
        console.log("comment:", comment);

        // make sure some fields are filled
        if (instance_quality == undefined && comment == "") {
            show_alert("No feedback is provided.", "danger", insert_after_selector="#feedback-submit", timeout=5000);
            return;
        }
        const response = await fetch("/api/submit-feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                index: current_index,
                model_a,
                model_b,
                prompt,
                completion_a,
                completion_b,
                instance_quality,
                comment,
                evaluator: username
            }),
        });

        // if the response is 200, show the success message
        if (response.status == 200) {
            show_alert("Feedback is submitted successfully.", "success", insert_after_selector="#feedback-submit", timeoutput=5000);
            console.log("Feedback is submitted successfully.");
        }
        else if (response.status == 401) {
            show_alert("You need to log in to submit feedback.", "danger", insert_after_selector="#feedback-submit", timeoutput=5000);
        }
        else {
            console.log(response);
            show_alert("Error when submitting feedback data. Please try again.", "danger", insert_after_selector="#feedback-submit", timeoutput=5000);
            console.error("Error when submitting feedback data:", response.status);
        }
    } catch (error) {
        show_alert("Error when submitting feedback data. Please try again.", "danger", insert_after_selector="#feedback-submit", timeoutput=5000);
        console.error("Error when submitting evaluation data:", error);
    }
}

$("#feedback-submit").click(function () {
    // prevent default form submission
    event.preventDefault();
    submit_feedback();
});

// Add event listeners for the navigation buttons
$('#prev-button').click(function () {
    if (current_index > 0) {
        // redirect to the previous instance using url
        window.location.href = `/instances/${current_index - 1}`;
    } else {
        show_alert("You are already on the first instance.", "danger");
    }
});

$("#next-button").click(function () {
    // redirect to the next instance using url
    window.location.href = `/instances/${current_index + 1}`;
});
