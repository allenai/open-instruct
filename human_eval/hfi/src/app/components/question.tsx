
export type SingleSelectQuestionParams = { 
  id: string,
  question: string, 
  description?: string,
  options: {
    label: string,
    value: string
  }[]
  selectedValue?: string
  onValueChanged?: (value: string) => void
}

export function SingleSelectQuestion({ id, question, description, options, selectedValue, onValueChanged } : SingleSelectQuestionParams) {

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
            name={id}
            value={option.value}
            defaultChecked={option.value === selectedValue}
            onChange={() => {
              if (typeof onValueChanged === 'function') {
                onValueChanged(option.value)
              }
            }}
          />
          <label className="form-check-label" htmlFor={`option-${index}`}>
            {option.label}
          </label>
        </div>
      ))}
    </div>
  )
}