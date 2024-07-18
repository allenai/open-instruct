import ChatMessage from "../components/chat-message";

export default function ModelOutput() {
  return (
    <div id="model-outputs-region" className="flex flex-col m-4 p-4 rounded w-full">
       <ChatMessage icon="🤖" message="Here are some responses from two AI models." />
      <div className="flex flex-col lg:flex-row my-4">
          <div className="flex flex-col lg:w-1/2">
              <div className="d-flex justify-content-center text-center">
                  <button className="completion-icon">A</button>
              </div>
          
              <div className="col completion-col rounded" id="completion-A-col">
                  {/* <xmp className="message-text">Loading model outputs ... </xmp> */}
              </div>
          </div>
          <div className="flex flex-col lg:w-1/2">
              <div className="d-flex justify-content-center text-center">
                  <button className="completion-icon">B</button>
              </div>
              <div className="col completion-col rounded" id="completion-B-col">
                  {/* <xmp className="message-text">Loading model outputs ... </xmp> */}
              </div>
          </div>
      </div>
  </div>
  )
}