export default function ChatMessage({ icon, message, children }: { icon: string, message: string, children?: React.ReactNode }) {

  let Children = () => {
    if (!children) {
      return <></>
    }

    return (
      <>
        <div className="flex mt-2">
            {children}
        </div>
      </>
    )
  }

  return (
    <div className="flex-col w-full">
      <div className="flex flex-col sm:flex-row h-fit w-full">
        <div className="h-12 w-12 m-auto">
          <button className="role-icon h-12 w-12 aspect-square">{icon}</button>
        </div>
        <p className="message-text m-4 sm:m-4">
            {message}
        </p>
      </div>
      <Children />
    </div>
  )
}

// Some instruction and input.