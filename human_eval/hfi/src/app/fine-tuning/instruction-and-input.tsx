import ChatMessage from "../components/chat-message"

export default function InstructionAndInput() {
  return (
    <div id="history-message-region" className="flex m-4 p-4 rounded w-full">
        <div className="flex flex-col sm:flex-row h-fit w-full">
          <ChatMessage icon="🧑" message="Some instruction and input">
            <p>Children information here</p>
            </ChatMessage>
        </div>
    </div>
  )
}