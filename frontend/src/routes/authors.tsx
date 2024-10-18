import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'
import { getAuthorsAuthorsGet } from '../generated_types'

export const Route = createFileRoute('/authors')({
  component: AuthorsComponent,
})

function AuthorsComponent() {

  const query = useQuery({
    queryKey: ['authors'],
    queryFn: () =>
      getAuthorsAuthorsGet(),
  })
  if (query.isLoading) {
    return <div>Loading...</div>
  }

  return <div>
    {query.data?.data?.map(([author_name, author_id, num_works]) => (
      <div key={author_id}>
        <Link to={'/author/$authorId'} params={{authorId: author_id}}>{author_name}: {num_works} papers checked</Link>
      </div>
    ))}
  </div>
  // return <div>Hello /authors!</div>
}
