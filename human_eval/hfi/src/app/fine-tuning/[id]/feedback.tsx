export default function Feedback() {
  return (
    <div id="nav" className="w-2/6">
        <h2 className="text-center" id="title"><a href="/">🕵 Human Evaluation</a></h2> 
        <div className="row mb-4" id="login-info">
            <div className="col-12 text-center">
                
                {/* {% if current_user.is_authenticated %}
                    You are logged in as <b>{{ current_user.username }}</b>. [<a href="{{ url_for('logout') }}">Log out</a>]
                {% else %}
                    You need to log in to submit evaluation. [<a href="{{ url_for('login') }}">Log in</a>]
                {% endif %} */}
            </div>
        </div>

        <div className="flex flex-row m-4">
            <div className="w-fit text-center">
                <button id="prev-button" className="btn btn-sm btn-primary"> &lt; </button>
            </div> 
            <div className="w-full text-center">
                <h4 id="instance-id" className="text-center">Instance n</h4>
            </div>
            <div className="w-fit text--center">
                <button id="next-button" className="btn btn-sm btn-primary"> &gt; </button>
            </div>
        </div>

        <form>
            <div className="form-group">
                <p>Do you find the instance interesting, invalid, or too hard to complete? Please let us know by giving feedback here! (Optional)</p>
                <div className="form-check form-check-inline">
                    <input className="form-check-input" type="radio" name="instance-quality" id="instance-quality-good" value="good" />
                    <label className="form-check-label" htmlFor="instance-quality-good">This example is interesting.</label>
                </div>
                <div className="form-check form-check-inline">
                    <input className="form-check-input" type="radio" name="instance-quality" id="instance-quality-bad" value="bad" />
                    <label className="form-check-label" htmlFor="instance-quality-bad">This example is invalid.</label>
                </div>
                <div className="form-check form-check-inline">
                    <input className="form-check-input" type="radio" name="instance-quality" id="instance-quality-hard" value="hard" />
                    <label className="form-check-label" htmlFor="instance-quality-bad">This example is too hard for me.</label>
                </div>
            </div>
            <div className="form-group mt-4">
                <label htmlFor="comment">Comment:</label>
                <textarea className="form-control" id="comment" name="comment" rows={4}></textarea>
            </div>
            <div className="mt-2 text-center">
                <button id="feedback-submit" type="submit" className="btn btn-primary mt-2">Provide feedback</button>
            </div>
        </form>
    </div>
  )
}