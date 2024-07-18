
export type SingleSelectQuestionParams = { 
  question: string, 
  description?: string,
  options: {
    label: string,
    value: string
  }[]
}

export function SingleSelectQuestion({ question, description, options } : SingleSelectQuestionParams) {

  const Description = () => {
    if (!description) {
      return <></>
    }

    return (
      <p className="text-sm text-gray-400 mb-2">{description}</p>
    )
  }

  return (
    <div className="form-group eval-form-item">
        <p className="text-base font-semibold mb-2">{question}</p>
        <Description />
        {options.map((option, index) => (
        <div key={index} className="form-check form-check-inline my-2">
          <input
            className="form-check-input m-2"
            type="radio"
            name="single-select"
            id={`option-${index}`}
            value={option.value}
          />
          <label className="form-check-label" htmlFor={`option-${index}`}>
            {option.label}
          </label>
        </div>
      ))}
    </div>
  )
}